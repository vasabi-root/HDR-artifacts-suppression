import torch
import torch.nn as nn
import torch.functional as F
from efficientvit.models.nn import EfficientViTBlock, ResidualBlock, MBConv, IdentityLayer
from model.efficient_vit import lightMSLA
from torchsummary import summary
from model.gfu.guided_filter import FastGuidedFilter
from losses.gmef_ssim import clamp_min

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

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
    
class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.in_norm = nn.InstanceNorm2d(n, affine=True, track_running_stats=False)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.in_norm(x)

class LUTMEF(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_exposures = num_exposures
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, 1, 1),
            nn.ReLU(),
            self.norm,
        )
        self.local_attention = nn.ModuleList([
            nn.Sequential(
                EfficientViTBlock(embed_dim),
                nn.ReLU()
            )
            for _ in range(self.num_exposures)
        ])

        self.local_attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 256)),
                nn.Conv2d(embed_dim, 1, 1, 1, 0), 
                nn.Sigmoid()               
            )
            for _ in range(self.num_exposures)
        ])

        

        



        

class WeightsExtractor(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_exposures = num_exposures
        self.embed_dim = embed_dim
        
        self.norm = AdaptiveNorm(embed_dim)
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, 1, 1),
            nn.ReLU(),
            self.norm,
        )
        self.global_attention = EfficientViTBlock(embed_dim)
        
        # Блоки внимания на основе EfficientViT
        self.motion_attention = nn.ModuleList([
            # nn.Sequential(
            EfficientViTAttention(dim=embed_dim)
                # nn.Dropout2d(),   
            # )
            for _ in range(num_exposures-1)
        ])
        
        # Слои для слияния признаков
        self.tails = nn.ModuleList([
            nn.Sequential(
                self.norm,
                nn.Conv2d(embed_dim, 1, 3, 1, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_exposures)
        ])
    


    def forward(self, exposures):
        # exposures: [B, num_exposures, C, H, W]
        batch_size, num_exposures, C, H, W = exposures.size()
        assert num_exposures == self.num_exposures

        # Выбираем опорный кадр (например, средний по экспозиции)
        ref_idx = num_exposures // 2
        ref = exposures[:, ref_idx]

        features_bunch = []
        for i in range(num_exposures):
            encoded = self.encoder(exposures[:, i]) 
            features = self.global_attention(encoded)
            features_bunch.append(features)
        
        # Извлечение признаков опорного кадра
        ref_features = features_bunch[ref_idx] # [B, feature_dim, H, W]
        
        # Обработка неопорных кадров через блоки внимания
        attended_features = []
        for i in range(num_exposures):
            if i != ref_idx:
                other_features = features_bunch[i]
                # Применение EfficientViTAttention
                attended = self.motion_attention[i if i < ref_idx else i-1](ref_features, other_features)
                attended_features.append(attended)
            else:
                attended_features.append(ref_features)
        
        weights = []
        for light, motion, tail in zip(features_bunch, attended_features, self.tails):
            weights.append(tail(motion))

        return torch.stack(weights, dim=1)

class PairEVitHDR(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64):
        super().__init__()
        self.weights_extractor = WeightsExtractor(in_channels, num_exposures, embed_dim)
        self.gf = FastGuidedFilter(2,)
    
    def forward(self, exposures_lr, exposures_hr):
        weights_lr = self.weights_extractor(exposures_lr)
        lights_lr = exposures_lr[0, :, 0].unsqueeze(1)
        lights_hr = exposures_hr[0, :, 0].unsqueeze(1)
        weights_hr = self.gf(lights_lr, weights_lr[0], lights_hr).unsqueeze(0)

        output_hr = (weights_hr*exposures_hr).sum(1) / clamp_min(weights_hr.sum(1))
        return output_hr, weights_hr
        
# Пример использования
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PairEVitHDR(in_channels=3, num_exposures=3, embed_dim=32).to(device)
    sample_input_lr = torch.randn(1, 3, 3, 256, 356).to(device)
    sample_input_hr = torch.randn(1, 3, 3, 1500, 2000).to(device)
    output, _ = model(sample_input_lr, sample_input_hr)
    # print(output.shape)
    summary(model, [sample_input_lr.shape[1:],sample_input_hr.shape[1:]], 1)