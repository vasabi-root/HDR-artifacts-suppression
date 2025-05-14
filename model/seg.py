import torch
import torch.nn as nn
import torch.nn.functional as F

from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone
from efficientvit.models.efficientvit.seg import (
    EfficientViTBackbone,
    SegHead,
    EfficientViTSeg,
)
from model.gfu.guided_filter import FastGuidedFilter
from model.efficient_vit import lightMSLA
from efficientvit.models.nn import EfficientViTBlock, ResidualBlock, MBConv, IdentityLayer
from efficientvit.models.utils import build_kwargs_from_config
from data import BunchDataset, BracketingBunch


import numpy as np
from pathlib import Path
import os

def efficientvit_seg_b2(num_exposures=3, **kwargs) -> EfficientViTSeg:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1

    backbone = efficientvit_backbone_b1(**kwargs)
    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[256, 128, 64],
        stride_list=[32, 16, 8],
        head_stride=8,
        head_width=64,
        head_depth=3,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=num_exposures,
        **build_kwargs_from_config(kwargs, SegHead),
    )

    model = EfficientViTSeg(backbone, head)
    return model

def efficientvit_seg_b0(num_exposures=3, **kwargs) -> EfficientViTSeg:
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0

    backbone = efficientvit_backbone_b0(**kwargs)

    head = SegHead(
        fid_list=["stage4", "stage3", "stage2"],
        in_channel_list=[128, 64, 32],
        stride_list=[32, 16, 8],
        head_stride=1,
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op="mbconv",
        final_expand=4,
        n_classes=num_exposures,
        **build_kwargs_from_config(kwargs, SegHead),
    )

    model = EfficientViTSeg(backbone, head)
    return model

def load_pretrained_backbone(backbone_model: EfficientViTBackbone, weights_path: Path):
    d = torch.load(weights_path)
    backbone_dict = {key.replace('backbone.', '') :d['state_dict'][key] for key in d['state_dict'].keys() if 'backbone.' in key}
    backbone_dict.pop('input_stem.op_list.0.conv.weight')
    backbone_model.load_state_dict(backbone_dict, strict=False)

class EfficientViTAttention(nn.Module):
    def __init__(self, dim):
        super(EfficientViTAttention, self).__init__()
        self.dim = dim

        # Механизм cross-attention
        self.attention = lightMSLA(dim, dim)
    
        # Проекционные слои для query, key, value
        self.proj_q = nn.Conv2d(dim, self.attention.total_dim, 3, 1, 1)
        self.proj_k = nn.Conv2d(dim, self.attention.total_dim, 3, 1, 1)
        self.proj_v = nn.Conv2d(dim, self.attention.total_dim, 3, 1, 1)

        self.local_module = ResidualBlock(
            MBConv(
                in_channels=dim,
                out_channels=dim,
                # expand_ratio=6,
                use_bias=(True, True, False)
            ),
            IdentityLayer(),
        )


    def forward(self, ref_features, other_features):
        # ref_features: [B, C, H, W] - признаки опорного кадра
        # other_features: [B, C, H, W] - признаки неопорного кадра
        
        # Проекции для query (опорный кадр), key и value (неопорный кадр)
        query = self.proj_q(ref_features)
        key = self.proj_k(other_features)
        value = self.proj_v(other_features)
        
        # Cross-attention между опорным и неопорным кадрами
        attended = self.attention(torch.cat([query, key, value], 1))
        
        # Residual connection и нормализация
        attended = ref_features + attended
        output = self.local_module(attended)

        return output

class SpatialAttentionModule(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.att1 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = self.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map

class GhostsDetector(nn.Module):
    def __init__(self, num_exposures=3, channels=1, weights_path=None):
        super().__init__()
        self.num_exposures = num_exposures
        self.channels = channels
        self.vit = efficientvit_seg_b0(num_exposures=num_exposures, in_channels=num_exposures*channels)
        if weights_path:
            load_pretrained_backbone(self.vit.backbone, weights_path)
        self.gf = FastGuidedFilter(2,)
        self.ref_idx = num_exposures // 2

    def forward(self, exposures):
        assert len(exposures.shape) == 4
        assert exposures.shape[0] == self.num_exposures
        assert exposures.shape[1] == self.channels
        K, C, H, W = exposures.shape
        x = torch.cat([tensor for tensor in exposures], 0).unsqueeze(0)
        masks = self.vit(x)

        masks = masks.permute(1, 0, 2, 3)
        # masks_low = masks[:self.ref_idx]
        # mask_middle = torch.zeros((1, 1, H, W)).to(masks.device)
        # masks_high = masks[self.ref_idx:]
        # masks = torch.cat([masks_low, mask_middle, masks_high], 0)
        masks[masks <= 0.5] = 0
        masks[masks > 0.5] = 1

        masks = 1.0 - masks
        return masks

def get_weight_map(luts_dir: Path):
    num = len(os.listdir(luts_dir))
    fs = [open(os.path.join(luts_dir, str(i) + "_weight.txt"), 'r') for i in range(num)]
    data = [f.read().split(',') for f in fs]
    weight_map = np.zeros([num, 256], np.double)
    for k in range(num):
        for j in range(256):
            d = float(data[k][j])
            if d > 1.0:
                d = 1.0
            if d < 0.0:
                d = 0.0
            weight_map[k][j] = d
    [f.close() for f in fs]
    return weight_map

def get_fusion_mask(small_imgs, weight_map, img_masks):
    for k in range(small_imgs.shape[0]):
        img_masks[k:k+1] = weight_map[k:k+1][:, small_imgs[k:k+1]]
    return img_masks.squeeze(0)

if __name__ == '__main__':
    from torchsummary import summary
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\results\sequences\17'))
    backbones_dir = Path(r'efficientvit-seg')
    model = GhostsDetector(num_exposures=3, channels=3, weights_path=backbones_dir / 'efficientvit_seg_b0_cityscapes.pt')
    model.to(device)
    model.train()
    out = model(torch.rand((3, 3, 1280, 1664)).to(device))
    # summary(model.to('cuda:0'), (3, 1500, 2000), 1)
    pass
    