import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from losses.plot import plot_weights, plot_imgs
from losses.mef_ssim import mef_ssim, create_window
from torchvision.io import decode_image

def clamp_min(tensor, minval=1e-6):
    return torch.clamp(tensor, min=minval)


def ssim(x, y, window_size=11, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=window_size//2)
    
    sigma_x = F.avg_pool2d(x**2, window_size, stride=1, padding=window_size//2) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, window_size, stride=1, padding=window_size//2) - mu_y**2
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=window_size//2) - mu_x * mu_y
    
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = clamp_min((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    
    return (numerator / (denominator + 1e-8)).mean()

def create_grad_window(window_size, channel):
    distances_x_1D = torch.linspace(-(window_size//2), window_size//2, window_size)
    distances_x = distances_x_1D.expand((window_size, window_size))
    distances_y = distances_x.T

    kernel_x = distances_x / (distances_x**2 + distances_y**2 + 1e-6)
    kernel_y = kernel_x.T

    kernel_x = kernel_x.expand(1, channel, window_size, window_size).float().contiguous()
    kernel_y = kernel_y.expand(1, channel, window_size, window_size).float().contiguous()

    return kernel_x, kernel_y


def calc_grad(img, window_size):
    kernel_x, kernel_y = create_grad_window(window_size, img.shape[1])
    kernel_x = kernel_x.to(img.get_device())
    kernel_y = kernel_y.to(img.get_device())

    grad_x = F.conv2d(img, kernel_x, stride=1, padding=window_size//2)
    grad_y = F.conv2d(img, kernel_y, stride=1, padding=window_size//2)
    grad_x = torch.clamp(grad_x, -1, 1)
    grad_y = torch.clamp(grad_y, -1, 1)

    grad_abs = torch.sqrt(clamp_min(grad_x**2 + grad_y**2, minval=1e-8))
    grad_ang = torch.atan(grad_y / clamp_min(grad_x))

    return grad_abs, grad_ang


def calc_grad_ssim(output, bunch, window_size):
    K, C, H, W = bunch.shape
    grad_Y_abs, grad_Y_ang  = calc_grad(bunch, window_size)
    grad_X_abs, grad_X_ang  = calc_grad(output, window_size)
    grad_X_sq_abs, grad_X_sq_ang  = calc_grad(output**2, window_size)
    grad_Y_sq_abs, grad_Y_sq_ang  = calc_grad(bunch**2, window_size)
    grad_XY_abs, grad_XY_ang  = calc_grad(output * bunch, window_size)

    grad_X_cov_abs = clamp_min(grad_X_sq_abs - grad_X_abs**2)
    grad_Y_cov_abs = clamp_min(grad_Y_sq_abs - grad_Y_abs**2)
    grad_XY_cov_abs = clamp_min(grad_XY_abs - grad_X_abs * grad_Y_abs)

    grad_X_cov_ang = clamp_min(grad_X_sq_ang - grad_X_ang**2)
    grad_Y_cov_ang = clamp_min(grad_Y_sq_ang - grad_Y_ang**2)
    grad_XY_cov_ang = clamp_min(grad_XY_ang - grad_X_ang * grad_Y_ang)

    grad_covs_abs = (2 * grad_XY_cov_abs) / clamp_min(grad_X_cov_abs + grad_Y_cov_abs)
    grad_covs_ang = (2 * grad_XY_cov_ang) / clamp_min(grad_X_cov_ang + grad_Y_cov_ang)

    grad_means_abs = (2*grad_X_abs*grad_Y_abs / clamp_min(grad_X_abs**2 + grad_Y_abs**2))
    grad_means_ang = (2*grad_X_ang*grad_Y_ang / clamp_min(grad_X_ang**2 + grad_Y_ang**2))

    grad_ssim_abs = grad_means_abs * grad_covs_abs
    grad_ssim_ang = grad_means_ang * grad_covs_ang
    
    grad_ssim_abs = torch.clamp(grad_ssim_abs, 0, 1)
    grad_ssim_ang = torch.clamp(grad_ssim_ang, 0, 1)
    
    # plot_imgs(grad_ssim_abs.view(H, W), grad_ssim_ang.view(H, W))

    return grad_ssim_abs, grad_ssim_ang


def _gmef_ssim(output_wmotion, output, bunch, ws):
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "NaN/Inf in output"
    assert not torch.isnan(output_wmotion).any() and not torch.isinf(output_wmotion).any(), "NaN/Inf in output"
    assert not torch.isnan(bunch).any() and not torch.isinf(bunch).any(), "NaN/Inf in bunch"

    # bunch: [K C H W]
    # output: [C H W]
    K, C, H, W = bunch.shape

    window = create_window(ws, C)
    mefssim, patch_index = mef_ssim(output_wmotion, bunch, ws, is_lum=True)

    ref_idx = len(bunch) // 2
    ref = bunch[ref_idx].unsqueeze(0)
    grad_ssim_abs, grad_ssim_ang = calc_grad_ssim(output, bunch, ws)
    
    ref_mean = ref.mean(1)
    mask = ((ref_mean > 1/32) & (ref_mean < 31/32))
    grad_ssim_abs_ref_m = clamp_min(grad_ssim_abs[ref_idx][mask].mean())
    grad_ssim_ang_ref_m = clamp_min(grad_ssim_ang[ref_idx][mask].mean())

    # grad_ssim_abs_patch = torch.gather(grad_ssim_abs.view(K, C, -1), 0, patch_index.expand((C, H, W)).view(1, C, -1)).view(C, H, W)
    # grad_ssim_ang_patch = torch.gather(grad_ssim_ang.view(K, C, -1), 0, patch_index.expand((C, H, W)).view(1, C, -1)).view(C, H, W)
    grad_ssim_abs_patch = torch.gather(grad_ssim_abs.view(K, -1), 0, patch_index.view(1, -1)).view(1, H, W)
    grad_ssim_ang_patch = torch.gather(grad_ssim_ang.view(K, -1), 0, patch_index.view(1, -1)).view(1, H, W)

    grad_ssim_abs_patch_m = 1.0
    grad_ssim_ang_patch_m = 1.0
    anti_mask = ~mask
    if len(anti_mask) > 0 and (anti_mask).all() != False:
        grad_ssim_abs_patch_m = clamp_min(grad_ssim_abs_patch[anti_mask].mean())
        grad_ssim_ang_patch_m = clamp_min(grad_ssim_ang_patch[anti_mask].mean())

    grad_ssim_abs_m = clamp_min(grad_ssim_abs_ref_m * grad_ssim_abs_patch_m)
    grad_ssim_ang_m = clamp_min(grad_ssim_ang_ref_m * grad_ssim_ang_patch_m)

    # grad_ssim_abs[ref_idx][~mask] = 0.0
    # grad_ssim_ang[ref_idx][~mask] = 0.0
    # grad_ssim_abs_patch[mask] = 0.0
    # plot_weights(bunch.permute(0, 2, 3, 1), grad_ssim_abs[ref_idx].mean(0), grad_ssim_abs_patch.mean(0))

    metric = mefssim * grad_ssim_abs_m * grad_ssim_ang_m
    assert not torch.isnan(metric)

    return clamp_min(metric)
    
def gmef_ssim(output_wmotion, output, bunch, ws):
    # beta = torch.Tensor([0.0710, 0.4530, 0.4760])
    # beta = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    # beta = torch.Tensor([1, 1, 1, 1, 1])
    beta = torch.Tensor([1])
    if output.get_device() >= 0:
        beta = beta.to(output.get_device())

    levels = beta.size()[0]
    vals = []
    for _ in range(levels):
        val = _gmef_ssim(output_wmotion, output, bunch, ws)
        vals.append(val)

        output_wmotion = F.avg_pool2d(output_wmotion, (2, 2))
        output = F.avg_pool2d(output, (2, 2))
        bunch = F.avg_pool2d(bunch, (2, 2))

    vals = torch.stack(vals)
    return torch.prod(vals ** beta)

class GMEF_MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, output_wmotion, output, bunch):
        return gmef_ssim(output_wmotion, output, bunch, self.window_size)
    

if __name__ == '__main__':
    from data import BracketingBunch
    from pathlib import Path
    # import numpy as np
    import cv2

    output = decode_image(r"D:\windows\Documens\Diploma\results\fused\fused6_1_scale_grad_all_lum_100_ep\17.jpg") / 255
    C, W, H = output.shape
    output = F.interpolate(output.unsqueeze(0), [720, int(H/W*720)], mode='bilinear', align_corners=True)
    bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\dataset\test\test\17'))

    bunch_tensor = []
    for img in bunch:
        tensor = torch.tensor(F.interpolate(img.unsqueeze(0), (output.shape[2], output.shape[3]))) / 255
        bunch_tensor.append(tensor)
    bunch = torch.cat(bunch_tensor)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output = output.to(device)
    bunch = bunch.to(device)
    loss_fn = GMEF_MSSSIM()
    for _ in range(10):
        loss = loss_fn(output, output, bunch)
        print(loss)