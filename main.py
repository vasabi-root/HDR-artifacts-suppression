import cv2
import numpy as np
import os
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from model.efficient_hdr_light import EfficientHDR
from model.pair_evit_hdr_wg import PairEVitHDR
from model.e2emef import E2EMEF

import torch
from data import BunchDataset, RGBToYCbCr, YCbCrToRGB
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from sklearn.model_selection import train_test_split
from trainer_ys import Trainer


def main():
    num_exposures = 3
    transform_lr = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(700),
        RGBToYCbCr(),
    ])

    transform_hr = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize(700),
        RGBToYCbCr(),
    ])

    transform_out = YCbCrToRGB()
    
    train_set = BunchDataset(
        r'D:\windows\Documens\Diploma\dataset\old\aligned', 
        num_exposures=num_exposures, 
        transform_lr=transform_lr,
        transform_hr=transform_hr,
    )

    # train_set, val_set = train_test_split(dataset, train_size=0.8)
    test_set = BunchDataset(
        r'D:\windows\Documens\Diploma\dataset\test\test', 
        num_exposures=num_exposures, 
        transform_lr=transform_lr,
        transform_hr=transform_hr,
    )

    train_loader = DataLoader(train_set, 1, num_workers=3)
    val_loader = DataLoader(test_set, 1, num_workers=3)
    test_loader = DataLoader(test_set, 1, num_workers=3)
    
    # model = PairEVitHDR(
    #     in_channels=3, 
    #     num_exposures=num_exposures, 
    #     embed_dim=32
    # )

    model = E2EMEF()

    name = 'MEFnet'
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
        load_from_disk=True,
    )


if __name__ == '__main__':
    main()