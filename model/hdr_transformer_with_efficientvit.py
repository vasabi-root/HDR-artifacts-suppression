import torch
import torch.nn as nn
import torch.functional as F
from efficientvit.models.nn import EfficientViTBlock
from model.efficient_vit import lightMSLA

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
    def __init__(self, dim, num_heads=2):
        super(EfficientViTAttention, self).__init__()
        
        # Инициализация EfficientViT как основы механизма внимания
        self.efficientvit = EfficientViTBlock(in_channels=dim)
        
        # Проекционные слои для query, key, value
    
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        
        # Механизм cross-attention
        self.attention = lightMSLA(dim, dim, heads=8)
        
        # Feed-Forward сеть для постобработки
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Нормализация слоев
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, ref_features, other_features):
        # ref_features: [B, C, H, W] - признаки опорного кадра
        # other_features: [B, C, H, W] - признаки неопорного кадра
        
        # Извлечение признаков через EfficientViT
        ref_vit = self.efficientvit(ref_features)  # [B, C', H', W']
        other_vit = self.efficientvit(other_features)  # [B, C', H', W']
        
        # Преобразование в формат [seq_len, B, C'], где seq_len = H'*W'
        B, C_prime, H_prime, W_prime = ref_vit.size()
        ref_flat = ref_vit.view(B, C_prime, -1).permute(2, 0, 1)  # [H'*W', B, C']
        other_flat = other_vit.view(B, C_prime, -1).permute(2, 0, 1)  # [H'*W', B, C']
        
        # Проекции для query (опорный кадр), key и value (неопорный кадр)
        query = self.proj_q(ref_flat)
        key = self.proj_k(other_flat)
        value = self.proj_v(other_flat)
        
        # Cross-attention между опорным и неопорным кадрами
        attended, _ = self.attention(query, key, value)
        
        # Residual connection и нормализация
        ref_flat = ref_flat + attended
        ref_flat = self.norm1(ref_flat)
        
        # Применение Feed-Forward сети
        ffn_out = self.ffn(ref_flat)
        ref_flat = ref_flat + ffn_out
        ref_flat = self.norm2(ref_flat)
        
        # Возвращаем результат в исходный формат [B, C', H', W']
        output = ref_flat.permute(1, 2, 0).view(B, C_prime, H_prime, W_prime)
        return output

class HDRTransformer(nn.Module):
    def __init__(self, in_channels=3, num_exposures=3, embed_dim=64):
        super(HDRTransformer, self).__init__()
        
        # Энкодер на основе сверточной сети
        self.encoders = [
            nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
            for _ in range(num_exposures)
        ]
        self.pairs_attention = [
            SpatialAttentionModule(embed_dim)
            for _ in range(num_exposures-1)
        ]
        self.conv_first = nn.Conv2d(embed_dim * num_exposures, embed_dim, 3, 1, 1)
        
        # Блоки внимания на основе EfficientViT
        self.attention_blocks = nn.ModuleList([
            EfficientViTAttention(dim=embed_dim) for _ in range(num_exposures - 1)
        ])
        
        # Слои для слияния признаков
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * (num_exposures - 1), embed_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, exposures):
        # exposures: [B, num_exposures, C, H, W]
        batch_size, num_exposures, C, H, W = exposures.size()
        
        # Выбираем опорный кадр (например, средний по экспозиции)
        ref_idx = num_exposures // 2
        ref = exposures[:, ref_idx]
        
        # Извлечение признаков опорного кадра
        ref_features = self.encoder(ref)  # [B, feature_dim, H, W]
        
        # Обработка неопорных кадров через блоки внимания
        attended_features = []
        for i in range(num_exposures):
            if i != ref_idx:
                other = exposures[:, i]
                other_features = self.encoder(other)
                # Применение EfficientViTAttention
                attended = self.attention_blocks[i if i < ref_idx else i-1](ref_features, other_features)
                attended_features.append(attended)
        
        # Объединение обработанных признаков
        fused_features = torch.cat(attended_features, dim=1)  # [B, feature_dim * (num_exposures - 1), H, W]
        
        # Генерация HDR-изображения
        output = self.fusion(fused_features)
        output = self.sigmoid(output)
        
        return output

# Пример использования
if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = HDRTransformer(in_channels=3, num_exposures=3, feature_dim=64).to(device)
    sample_input = torch.randn(1, 3, 3, 128, 128).to(device)
    output = model(sample_input)
    print(output.shape)