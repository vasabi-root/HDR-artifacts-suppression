import os
from pathlib import Path

import numpy as np
import cv2
import torch

def save_image_tensor(tensor: torch.Tensor, name: Path):
    img = tensor.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))