import cv2
import numpy as np
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from model.efficient_hdr_light import EfficientHDR
from model.pair_evit_hdr_wg import PairEVitHDR
from model.seg import GhostsDetector
from model.e2emef import E2EMEF

import torch
from data import BunchDataset, RGBToYCbCr, YCbCrToRGB, BunchWeightedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.model_selection import train_test_split
from trainer_seg import Trainer


def main():
    num_exposures = 3
    transform_lr = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(700),
        RGBToYCbCr(),
    ])

    # inference_size = (960, 1280)
    inference_size = (576, 768)

    transform_exp = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(inference_size),
        RGBToYCbCr(),
    ])

    transform_w = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(inference_size),
    ])

    transform_out = YCbCrToRGB()
    
    train_set = BunchWeightedDataset(
        # r'D:\windows\Documens\Diploma\dataset\train\train', 
        # r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep\train\weights',
        r'D:\windows\Documens\Diploma\dataset\kalantari\Training',
        r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep_kalantari\train\weights',
        num_exposures=num_exposures, 
        transform_exposure=transform_exp,
        transform_w=transform_w,
    )

    # train_set, val_set = train_test_split(dataset, train_size=0.8)
    test_set = BunchWeightedDataset(
        # r'D:\windows\Documens\Diploma\dataset\test\test', 
        # r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep\test\weights',
        r'D:\windows\Documens\Diploma\dataset\kalantari\Testing',
        r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep_kalantari\test\weights',
        num_exposures=num_exposures, 
        transform_exposure=transform_exp,
        transform_w=transform_w,
    )

    train_loader = DataLoader(train_set, 1, num_workers=3)
    val_loader = DataLoader(test_set, 1, num_workers=3)
    test_loader = DataLoader(test_set, 1, num_workers=3)
    
    # model = PairEVitHDR(
    #     in_channels=3, 
    #     num_exposures=num_exposures, 
    #     embed_dim=32
    # )
    backbones_dir = Path(r'efficientvit-seg')
    weights_path=backbones_dir / 'efficientvit_seg_b0_cityscapes.pt'
    model = GhostsDetector(
        num_exposures=num_exposures,
        channels=1,
        weights_path=weights_path
    )

    name = 'GhostDetector'
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        test_loader=test_loader,
        name=name,
        save_ckpt_dir=Path('./checkpoints') / name,
        save_res_dir=Path('./results') / name,
        transform_out=transform_out,
    )

    trainer.train(
        epoch_num=100,
        epochs_per_val=5,
        save_val=True,
        load_from_disk=False,
    )


if __name__ == '__main__':
    main()