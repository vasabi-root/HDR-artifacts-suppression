import torch
import torch.nn.functional as F
from math import exp
from utils.plot import plot_weights, plot_imgs

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / (gauss.sum())

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size/6.).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(1, channel, window_size, window_size).contiguous()) / channel
    return window

def create_uniform_window(window_size, channel):
    window = torch.ones((1, channel, window_size, window_size)).float().contiguous()
    window /= window.sum()
    return window


def _mef_ssim(output, bunch, window, ws, denom_g, denom_l, C1, C2, is_lum=False, is_variance_lum=False, full=False, full_map=False):
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "NaN/Inf in output"
    assert not torch.isnan(bunch).any() and not torch.isinf(bunch).any(), "NaN/Inf in bunch"

    K, C, H, W = list(bunch.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(bunch, window, padding=ws // 2).view(K, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(bunch * bunch, window, padding=ws // 2).view(K, H, W) \
                        - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image output
    muX = F.conv2d(output, window, padding=ws // 2).view(H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(output * output, window, padding=ws // 2).view(H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(output.expand_as(bunch) * bunch, window, padding=ws // 2).view(K, H, W) \
                - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2).clamp(min=1e-6)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)

    if is_variance_lum:
        # 1. Вычисляем эталонную яркость muY_ref (как в оригинале)
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY_ref = torch.sum(LY * muY_seq, dim=0) / torch.sum(LY, dim=0)

        # 2. Яркость выходного изображения (muX) и эталонной (muY_ref)
        # muX = F.conv2d(output, window, padding=ws//2).view(H, W)  # уже вычислено ранее
        
        # 3. Расчёт ковариации и дисперсий для яркости
        # Формируем расширенные тензоры для поэлементных операций
        muX_expanded = muX.unsqueeze(0)  # [1, H, W]
        muY_ref_expanded = muY_ref.unsqueeze(0)  # [1, H, W]
        
        # Ковариация между muX и muY_ref (по пространственным координатам)
        window_cov = create_window(11, 1).to(device='cuda:0' if torch.cuda.is_available() else 'cpu')
        cov_ly = F.conv2d(
            (muX_expanded * muY_ref_expanded).unsqueeze(0),  # [1, 1, H, W]
            window_cov, 
            padding=ws//2
        ).squeeze() - (muX * muY_ref)
        
        # Дисперсии muX и muY_ref
        var_muX = F.conv2d(muX_expanded.pow(2).unsqueeze(0), window_cov, padding=ws//2).squeeze() - muX.pow(2)
        var_muY = F.conv2d(muY_ref_expanded.pow(2).unsqueeze(0), window_cov, padding=ws//2).squeeze() - muY_ref.pow(2)
        
        # Яркостная карта через ковариацию и дисперсию
        l_map = ((2 * cov_ly + C1) / (var_muX + var_muY + C1).clamp(min=1e-6)).clamp(1e-6, 1)
    elif is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - 0.5) ** 2) / denom_l)
        lG = torch.exp(- ((lY - 0.5) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1).clamp(min=1e-6)
    else:
        l_map = torch.Tensor([1.0])
        if bunch.is_cuda:
            l_map = l_map.cuda(bunch.get_device())

    # gt_map = torch.gather(muY_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)
    # plot_imgs(output, muX, cs_map, l_map)
    if full_map:
        return l_map, cs_map, patch_index
    
    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs, patch_index

    qmap = l_map * cs_map
    q = qmap.mean()

    return q, patch_index


def mef_ssim(output, bunch, window_size=11, is_lum=False, full_map=False, is_variance_lum=False):
    (_, channel, _, _) = bunch.size()
    window = create_window(window_size, channel)

    if bunch.is_cuda:
        window = window.cuda(bunch.get_device())
    window = window.type_as(bunch)

    return _mef_ssim(output, bunch, window, window_size, 0.08, 0.08, 0.01**2, 0.03**2, is_lum, is_variance_lum, full_map=full_map)


def mef_msssim(output, bunch, window, ws, denom_g, denom_l, C1, C2, is_lum=False, is_variance_lum=False):
    # beta = torch.Tensor([0.0710, 0.4530, 0.4760])
    # beta = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    # beta = torch.Tensor([1, 1, 1, 1, 1])
    beta = torch.Tensor([1])
    if bunch.is_cuda:
        window = window.cuda(bunch.get_device())
        beta = beta.cuda(bunch.get_device())

    window = window.type_as(bunch)

    levels = beta.size()[0]
    l_i = []
    cs_i = []
    for _ in range(levels):
        l, cs, _ = _mef_ssim(output, bunch, window, ws, denom_g, denom_l, C1, C2, is_lum=is_lum, is_variance_lum=is_variance_lum, full=True)
        l_i.append(l)
        cs_i.append(cs)

        output = F.avg_pool2d(output, (2, 2))
        bunch = F.avg_pool2d(bunch, (2, 2))

    Ql = torch.stack(l_i)
    Qcs = torch.stack(cs_i)

    return torch.prod(Ql ** beta) * torch.prod(Qcs ** beta) 


class MEFSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEFSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, output, bunch):
        (_, channel, _, _) = bunch.size()

        if channel == self.channel and self.window.data.type() == bunch.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if bunch.is_cuda:
                window = window.cuda(bunch.get_device())
            window = window.type_as(bunch)

            self.window = window
            self.channel = channel

        return _mef_ssim(output, bunch, window, self.window_size,
                         self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)


class MEF_MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, c1=0.01, c2=0.03, is_lum=False):
        super(MEF_MSSSIM, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = create_window(window_size, self.channel)
        self.denom_g = 2 * sigma_g**2
        self.denom_l = 2 * sigma_l**2
        self.C1 = c1**2
        self.C2 = c2**2
        self.is_lum = is_lum

    def forward(self, output, bunch):
        (_, channel, _, _) = bunch.size()

        if channel == self.channel and self.window.data.type() == bunch.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if bunch.is_cuda:
                window = window.cuda(bunch.get_device())
            window = window.type_as(bunch)

            self.window = window
            self.channel = channel

        return mef_msssim(output, bunch, window, self.window_size,
                          self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum)