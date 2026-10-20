# Improving Scene Text Recognition with Counting-Aware Contrastive Learning and Attention Alignment

本仓库是 PRCV 2024 论文 **Improving Scene Text Recognition with Counting-Aware Contrastive Learning and Attention Alignment** 的代码。

训练与测试所用的数据集未包含在本仓库中，使用前请自行准备数据，并在配置文件里修改数据路径。字符集文件保留在 `Dino/data/charset_36.txt` 与 `Dino/data/charset_62.txt`。

## 环境

```bash
conda create -n CCD python=3.7
conda activate CCD
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
pip install tensorboard==1.15.0 tensorboardX==2.2
pip install -r requirement.txt
```

## 预训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --config ./Dino/configs/CCD_pretrain_ViT_Tiny.yaml
```

可选配置：`CCD_pretrain_ViT_small.yaml`、`CCD_pretrain_ViT_Base.yaml`。

## 微调

在 `CCD_vision_model_STD.yaml` 或 `CCD_vision_model_ARD.yaml` 中设置预训练权重路径后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_finetune.py --config ./Dino/configs/CCD_vision_model_STD.yaml
```

## 测试

在对应配置中设置微调权重路径后运行：

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config ./Dino/configs/CCD_vision_model_STD.yaml
```
