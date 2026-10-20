import header
import sys
import os
print(os.getcwd())
os.chdir("/home/yjj/CCD/")
sys.argv.extend(['--config', './Dino/configs/CCD_pretrain_ViT_Tiny.yaml'])
from train import main

if __name__ == '__main__':
    main()