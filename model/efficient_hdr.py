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
        
        # Извлечение признаков через EfficientViT
        # ref_vit = self.efficientvit(ref_features)  # [B, C', H', W']
        # other_vit = self.efficientvit(other_features)  # [B, C', H', W']
        
        # Преобразование в формат [seq_len, B, C'], где seq_len = H'*W'
        B, C_prime, H_prime, W_prime = ref_features.size()
        ref_flat = ref_features.view(B, C_prime, -1).permute(2, 0, 1)  # [H'*W', B, C']
        other_flat = other_features.view(B, C_prime, -1).permute(2, 0, 1)  # [H'*W', B, C']
        
        # Проекции для query (опорный кадр), key и value (неопорный кадр)
        query = self.proj_q(ref_features)
        key = self.proj_k(other_features)
        value = self.proj_v(other_features)
        
        # Cross-attention между опорным и неопорным кадрами
        attended = self.attention(torch.cat([query, key, value], 1))
        
        # Residual connection и нормализация
        other_features = ref_features + attended
        output = self.local_module(ref_features)

        return output

class EfficientHDR(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64, num_vit=4):
        super().__init__()
        self.in_channels = in_channels
        self.num_exposures = num_exposures
        self.embed_dim = embed_dim
        self.num_vit = num_vit

        self.ds_sequence = [
            Downsample(in_channels, in_channels)
            for _ in range(4)
        ]
        self.us_sequence = [
            Upsample(in_channels, in_channels)
            for _ in range(4)
        ]


        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, 3, 1, 1),
                nn.ReLU()
            )
            for _ in range(num_exposures)
        ])
        
        self.local_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
            )
            for _ in range(num_exposures)
        ]
        )

        self.conv_merged = nn.Sequential(
            nn.Conv2d(embed_dim * num_exposures, embed_dim * num_exposures, 3, 1, 1),
            nn.ReLU()
        )
        
        # Блоки внимания на основе EfficientViT
        self.global_attention = EfficientViTBlock(embed_dim*num_exposures)
        
        # Слои для слияния признаков
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * num_exposures, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def downsample(self, exposures):
        downsampled_sequences = [exposures]
        for i, ds in enumerate(self.ds_sequence):
            downsampled = []
            for j in range(self.num_exposures):
                dsed = ds(downsampled[i][:, j])
                downsampled.append(dsed)
            downsampled = torch.stack(downsampled)
            downsampled_sequences.append(downsampled)
        
        return downsampled_sequences
    
    def upsample(self, x, downsampled_sequences):
        upsampled_sequences = [exposures]
        for i, ds in enumerate(self.ds_sequence):
            downsampled = []
            for j in range(self.num_exposures):
                dsed = ds(downsampled[i][:, j])
                downsampled.append(dsed)
            downsampled = torch.stack(downsampled)
            downsampled_sequences.append(downsampled)
        
        return downsampled_sequences


    def forward(self, exposures):
        # exposures: [B, num_exposures, C, H, W]
        batch_size, num_exposures, C, H, W = exposures.size()
        assert num_exposures == self.num_exposures

        # Обработка неопорных кадров через блоки внимания
        attended_features = []
        for i in range(num_exposures):
            exposure = exposures[:, i]
            features = self.encoders[i](exposure)
            # Применение EfficientViTAttention
            attended = self.local_attention[i](features)
            attended_features.append(attended)
        
        # Объединение обработанных признаков
        merged_features = torch.cat(attended_features, dim=1)  # [B, feature_dim * (num_exposures - 1), H, W]
        merged_features = self.conv_merged(merged_features)
        merged_features_attended = self.global_attention(merged_features)
        
        # Генерация HDR-изображения
        output = self.fusion(merged_features_attended + merged_features)
        output = self.sigmoid(output)
        
        return output

# Пример использования
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = EfficientHDR(in_channels=3, num_exposures=3, embed_dim=32).to(device)
    sample_input = torch.randn(1, 3, 3, 720, 1280).to(device)
    output = model(sample_input)
    print(output.shape)
    summary(model, sample_input.shape[1:])