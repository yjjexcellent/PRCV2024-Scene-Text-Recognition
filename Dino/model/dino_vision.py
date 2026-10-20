import torch
import torch.nn as nn
import torch.nn.functional as F
import logging, time
import numpy as np
from fastai.vision import *

from Dino.modules import vision_transformer as vits
from torchvision import models as torchvision_models
from Dino.decoder.nrtr_decoder import NRTRDecoder
from Dino.convertor.attn import AttnConvertor
from Dino.loss.ce_loss import TFLoss

from Dino.modules.model_vision import BaseVision
from Dino.modules.model import Model, _default_tfmer_cfg
from Dino.modules.attention import *
from Dino.utils.DBSCAN import DBSCAN_cluster, label_cluster
dbscan = DBSCAN_cluster(eps=1.5, min_samples=4)
label = label_cluster()

class ABIDINOModel(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self,backbone,head):
        super(ABIDINOModel, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head
        self.len_classifier=My_lentoken_classifier()
        #self.Mapping = nn.AdaptiveAvgPool2d((out_instances, None))

    def generate_index_tensor_from_len(self,len_tensor):
        B=len_tensor.shape[0]
        max_length=26
        index_tensor=torch.zeros((B,max_length))
        for i,length in enumerate(len_tensor):
            index_tensor[i,:length]=1
        return index_tensor
    
    def forward(self,x,epoch,len_text):
        x1 = x[:, 1]    #色彩增强
        x2 = x[:, 2]    #几何增强 B 3 3 32 128 -> B 3 32 128
        # print(x1.shape,x2.shape)
        backbone_out = self.backbone(torch.cat([x1, x2]))#将两个视图拼到一起 2B 3 32 128
        N, T, E = backbone_out.shape #2B (32/4 128/4)+1 E
        # print(N,T,E) 2 257 192
        self.mask=nn.Parameter(torch.randn(N,26,T-1).cuda(),requires_grad=True)
        region_f = backbone_out[:,1:,:].reshape(N, 8, 32, E).permute(0, 3, 1, 2) #将宽高分离 2B E 8 32
        B=N//2
        len_token=backbone_out[:B,0,:]#2B E 这里只取第一个batch，也就是规则视图进行lentoken学习 B E
        # print(len_token.shape)
        len_token=self.len_classifier(len_token) # B
        # print(self.mask.device,region_f.device)
        # print(self.mask.shape,region_f.permute(0,2,3,1).reshape(N-1,-1,E).shape)
        attn_vecs=torch.bmm(self.mask,region_f.permute(0,2,3,1).reshape(N,-1,E)) #2B 26 E
        if epoch<10:#用gt长度分割特征 len_text B 
            index=self.generate_index_tensor_from_len(len_text) #B->B 26
        else:#用预测来进行分割  
            index=self.generate_index_tensor_from_len(len_token+1) #B->B 26
        index=torch.cat([index,index])#2B 26
        B=index.shape[0]
        length = torch.clamp(index.sum(1), 3, 26).unsqueeze(-1)[:B // 2]
        grid = torch.arange(0, 26, device=length.device).unsqueeze(0)
        new_index = grid <= length
        region_attention_out = torch.cat([attn_vecs[: B // 2][new_index], attn_vecs[B // 2:][new_index]])
        region_out=self.head(region_attention_out)
        res = {'instances_view': region_out,
                'feature': region_f,
                'len':len_token
                }
        return res

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DINO_Finetune(nn.Module):
    def __init__(self, config):
        super(DINO_Finetune, self).__init__()
        self.label_convertor = AttnConvertor(dict_type='DICT90', max_seq_len=config.decoder_max_seq_len, with_unknown=True)

        """ model configuration """
        # ============ building student and teacher networks ... ============
        # we changed the name DeiT-S for ViT-S to avoid confusions
        config.arch = config.arch.replace("deit", "vit")
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if config.arch in vits.__dict__.keys():
            self.backbone = vits.__dict__[config.arch](
                patch_size=config.patch_size,
                drop_path_rate=config.drop_path_rate,  # stochastic depth
            )
            embed_dim = self.backbone.embed_dim
        # if the network is a XCiT
        elif config.arch in torch.hub.list("facebookresearch/xcit:main"):
            self.backbone = torch.hub.load('facebookresearch/xcit:main', config.arch,
                                     pretrained=False, drop_path_rate=config.drop_path_rate)
            embed_dim = self.backbone.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif config.arch in torchvision_models.__dict__.keys():
            self.backbone = torchvision_models.__dict__[config.arch]()
            embed_dim = self.backbone.fc.weight.shape[1]
        else:
            print(f"Unknow architecture: {config.arch}")

        self.encoder = Mlp(in_features=embed_dim, hidden_features=512, out_features=512, act_layer=nn.GELU, drop=0.1)
        # self.encoder = None
        config.decoder_num_classes=self.label_convertor.num_classes()
        config.decoder_start_idx=self.label_convertor.start_idx
        config.decoder_padding_idx=self.label_convertor.padding_idx
        self.decoder = NRTRDecoder(
            n_layers=config.decoder_n_layers,
            d_embedding=config.decoder_d_embedding,
            n_head=config.decoder_n_head,
            d_k=config.decoder_d_k,
            d_v=config.decoder_d_v,
            d_model=config.decoder_d_model,
            d_inner=config.decoder_d_inner,
            n_position=200,
            dropout=0.1,
            num_classes=config.decoder_num_classes,
            max_seq_len=config.decoder_max_seq_len,
            start_idx=config.decoder_start_idx,
            padding_idx=config.decoder_padding_idx, )

        self.loss = TFLoss(ignore_index=self.label_convertor.padding_idx)

    def forward(self, img, text, return_loss=True, test_speed=False):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        """

        if return_loss:
            return self.forward_train(img, text)

        if test_speed:
            return self.forward_test_speed(img)

        return self.forward_test(img)

    def extract_feat(self, img):
        """Directly extract features from the backbone."""
        x, out = self.backbone(img)
        # x = x[:, 1:]
        return x

    def forward_train(self, img, img_metas):
        """
        Args:
            img (tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            dict[str, tensor]: A dictionary of loss components.
        """
        feat = self.extract_feat(img)

        # gt_labels = img_metas['text']
        # targets_dict = self.label_convertor.str2tensor(gt_labels)
        targets_dict = {'padded_targets': img_metas}

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec, attn = self.decoder(feat, out_enc, targets_dict, train_mode=True)

        loss_inputs = (out_dec, targets_dict,)
        losses = self.loss(*loss_inputs)

        return losses, attn

    def forward_test(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec = self.decoder(feat, out_enc, None, train_mode=False)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        # flatten batch results
        # results = []
        # for string, score in zip(label_strings, label_scores):
        #     results.append(dict(text=string, score=score))

        return out_dec

    def forward_test_speed(self, img):
        """Test function with test time augmentation.

        Args:
            imgs (torch.Tensor): Image input tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            list[str]: Text label result of each image.
        """
        feat = self.extract_feat(img)

        out_enc = None
        if self.encoder is not None:
            out_enc = self.encoder(feat)
        else:
            out_enc = feat

        out_dec = self.decoder(feat, out_enc, None, train_mode=False, test_speed=True)

        # early return to avoid post processing
        if torch.onnx.is_in_onnx_export():
            return out_dec

        # flatten batch results
        # results = []
        # for string, score in zip(label_strings, label_scores):
        #     results.append(dict(text=string, score=score))

        return out_dec
    

class My_lentoken_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm=nn.LayerNorm(192,eps=1e-6)
        self.fc1=nn.Linear(192,768)
        self.act_layer=nn.GELU()
        self.fc2=nn.Linear(768,26)

    def softargmax1d(self,input, beta=100):
        *_, n = input.shape
        input = nn.functional.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n)
        indices=indices.to(input.device)
        # print(type(indices),indices.device,input.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result
    
    def forward(self,x): #x 2B*192->2B 1
        x=self.layer_norm(x)
        x=self.fc1(x)
        x=self.act_layer(x)
        x=self.fc2(x) #此时x 2B 26
        x=self.softargmax1d(x)
        return x #2B