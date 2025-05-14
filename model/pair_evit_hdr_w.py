import torch
import torch.nn as nn
import torch.functional as F
from efficientvit.models.nn import EfficientViTBlock, ResidualBlock, MBConv, IdentityLayer
from model.efficient_vit import lightMSLA
from torchsummary import summary

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
        output = self.local_module(ref_features)

        return output

class PairEVitHDR(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64, num_vit=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_exposures = num_exposures
        self.embed_dim = embed_dim
        self.num_vit = num_vit
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embed_dim, 3, 1, 1),
            nn.ReLU(),
        )
        self.light_attention = EfficientViTBlock(embed_dim)
        
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
                nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1),
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

        light_features = []
        for i in range(num_exposures):
            encoded = self.encoder(exposures[:, i]) 
            features = self.light_attention(encoded)
            light_features.append(features)
        
        # Извлечение признаков опорного кадра
        ref_features = light_features[ref_idx] # [B, feature_dim, H, W]
        
        # Обработка неопорных кадров через блоки внимания
        attended_features = []
        for i in range(num_exposures):
            if i != ref_idx:
                other_features = light_features[i]
                # Применение EfficientViTAttention
                attended = self.motion_attention[i if i < ref_idx else i-1](ref_features, other_features)
                attended_features.append(attended)
            else:
                attended_features.append(ref_features)
        
        outputs = []
        for feature, tail in zip(attended_features, self.tails):
            outputs.append(tail(feature))
        
        return torch.stack(outputs, dim=1)

# Пример использования
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = PairEVitHDR(in_channels=3, num_exposures=3, embed_dim=32).to(device)
    sample_input = torch.randn(1, 3, 3, 300, 400).to(device)
    output = model(sample_input)
    print(output.shape)
    summary(model, sample_input.shape[1:])