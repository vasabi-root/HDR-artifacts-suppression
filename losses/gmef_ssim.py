import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
from utils.plot import plot_weights, plot_imgs, plot_bunches
from losses.mef_ssim import mef_ssim, create_window
from torchvision.io import decode_image

def clamp_min(tensor, minval=1e-6):
    return torch.clamp(tensor, min=minval)

def cut_region_from_bunch(bunch, x1, y1, x2, y2):
    regions = []
    for m in bunch:
        m = m.squeeze()
        if len(m.shape) == 2:
            regions.append(m[y1:y2, x1:x2])
            regions[-1][0][0] = 0.0
            regions[-1][0][1] = 1.0
        elif len(m.shape) == 3:
            regions.append(m[:, y1:y2, x1:x2])
        else:
            raise ValueError('The shape must be [C H W] or [H W]')
    return regions


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
    # plot_imgs(grad_ssim_abs[1].squeeze(), grad_ssim_ang[1].squeeze())
    # ref_idx = len(bunch)//2
    # full_maps = [bunch[ref_idx], output, grad_ssim_abs[ref_idx], grad_ssim_ang[ref_idx]]
    # right_ghosts_maps = cut_region_from_bunch(full_maps, 1115, 1622, 1310, 2121)
    # plot_bunches(full_maps, right_ghosts_maps, save_name=Path('plots/grad_maps.png'))

    return grad_ssim_abs, grad_ssim_ang


def _gmef_ssim(output, bunch, ws, return_map=False):
    assert not torch.isnan(output).any() and not torch.isinf(output).any(), "NaN/Inf in output"
    assert not torch.isnan(bunch).any() and not torch.isinf(bunch).any(), "NaN/Inf in bunch"

    # bunch: [K C H W]
    # output: [C H W]
    K, C, H, W = bunch.shape

    window = create_window(ws, C)
    l_map, cs_map, patch_index = mef_ssim(output, bunch, ws, is_lum=True, full_map=True)

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
    # anti_mask = ~mask
    if len(~mask) > 0 and (~mask).all() != False:
        grad_ssim_abs_patch_m = clamp_min(grad_ssim_abs_patch[~mask].mean())
        grad_ssim_ang_patch_m = clamp_min(grad_ssim_ang_patch[~mask].mean())

    grad_ssim_abs_m = clamp_min(grad_ssim_abs_ref_m * grad_ssim_abs_patch_m)
    grad_ssim_ang_m = clamp_min(grad_ssim_ang_ref_m * grad_ssim_ang_patch_m)

    grad_ssim_abs[ref_idx][~mask] = 0.0
    grad_ssim_ang[ref_idx][~mask] = 0.0
    grad_ssim_abs_patch[mask] = 0.0
    grad_ssim_ang_patch[mask] = 0.0
    grad_ssim_abs_full = grad_ssim_abs[ref_idx].clone(); grad_ssim_abs_full[~mask] = grad_ssim_abs_patch[~mask]
    grad_ssim_ang_full = grad_ssim_ang[ref_idx].clone(); grad_ssim_ang_full[~mask] = grad_ssim_ang_patch[~mask]
    grad_map = grad_ssim_ang_full.mean(0) #* grad_ssim_abs_full.mean(0)

    # plot_weights(bunch.permute(0, 2, 3, 1), grad_ssim_abs[ref_idx].mean(0), grad_ssim_abs_patch.mean(0))
    # plot_imgs(grad_ssim_abs[ref_idx].mean(0), grad_ssim_abs_patch.mean(0), grad_ssim_abs_full.mean(0))
    maps = [ref, output, grad_ssim_ang[ref_idx].mean(0), grad_ssim_ang_patch.mean(0), grad_ssim_ang_full.mean(0)]
    ghosts = cut_region_from_bunch(maps, 1115, 1622, 1310, 2121)
    plot_bunches(maps, ghosts, save_name=Path('plots/gssim.png'))
    # plot_imgs(cs_map, l_map, grad_map, cs_map*grad_map)

    # maps = [output.squeeze(), cs_map, l_map]
    # ghosts = []
    # halos = []
    # for m in maps:
    #     if len(m.shape) == 2:
    #         ghosts.append(m[1622:2121, 1115:1310])
    #         # halos.append(m[])
    #     else:
    #         ghosts.append(m[:, 1622:2121, 1115:1310])
    # plot_bunches(maps, ghosts)

    print(grad_ssim_ang_full.mean())
    mefssim = (l_map * cs_map).mean()

    metric = mefssim * grad_ssim_abs_m * grad_ssim_ang_m
    assert not torch.isnan(metric)

    if return_map:
        return l_map, cs_map, grad_map

    return clamp_min(metric)
    
def gmef_ssim(output, bunch, ws):
    # beta = torch.Tensor([0.0710, 0.4530, 0.4760])
    # beta = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    # beta = torch.Tensor([1, 1, 1, 1, 1])
    beta = torch.Tensor([1])
    if output.get_device() >= 0:
        beta = beta.to(output.get_device())

    levels = beta.size()[0]
    vals = []
    for _ in range(levels):
        val = _gmef_ssim(output, bunch, ws)
        vals.append(val)

        output = F.avg_pool2d(output, (2, 2))
        bunch = F.avg_pool2d(bunch, (2, 2))

    vals = torch.stack(vals)
    return torch.prod(vals ** beta)

class GMEF_MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, output, bunch):
        return gmef_ssim(output, bunch, self.window_size)
    

def test_mefssim():
    bunch_dir = Path(r'D:\windows\Documens\Diploma\results\sequences\105')
    output_path = Path(r'D:\windows\Documens\Diploma\src\results\test_mefssim\105_mertens.jpg')
    # output_path = Path(r'D:\windows\Documens\Diploma\src\results\test_mefssim\105_meflut.jpg')

    bunch = BracketingBunch(bunch_dir)
    output = decode_image(output_path).unsqueeze(0) / 255

    bunch_tensor = []
    for img in bunch:
        tensor = torch.tensor(F.interpolate(img.unsqueeze(0), (output.shape[2], output.shape[3]))) / 255
        bunch_tensor.append(tensor)
    bunch = torch.cat(bunch_tensor)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output = output.to(device)
    bunch = bunch.to(device)
    # loss_fn = GMEF_MSSSIM(5)
    # for _ in range(10):
    measure, _ = mef_ssim(output, bunch, is_lum=True)
    print(measure)




def test_meflut():
    output = decode_image(r"D:\windows\Documens\Diploma\results\fused\1dluts_eval\17.jpg") / 255
    # output = decode_image(r"D:\windows\Documens\Diploma\results\fused\04\Gauss.jpg") / 255
    C, W, H = output.shape
    # output = F.interpolate(output.unsqueeze(0), [1280, int(H/W*1280)], mode='bilinear', align_corners=True)
    output = output.unsqueeze(0)
    bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\results\sequences\17'))
    plots_dir = Path('plots')

    bunch_tensor = []
    for img in bunch:
        tensor = torch.tensor(F.interpolate(img.unsqueeze(0), (output.shape[2], output.shape[3]))) / 255
        bunch_tensor.append(tensor)
    bunch = torch.cat(bunch_tensor)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    output = output.to(device)
    bunch = bunch.to(device)

    cs_maps = []
    l_maps = []
    for ws in [3, 7, 11, 15]:
        l_map, cs_map, _ = _gmef_ssim(output, bunch, ws, return_map=True)
        cs_maps.append(cs_map)
        l_maps.append(l_map)

    output = output.squeeze()
    maps = [output.squeeze(), cs_map, l_map]
    ghosts = cut_region_from_bunch([output, *cs_maps], 1115, 1622, 1310, 2121)
    halos = cut_region_from_bunch([output, *cs_maps], 496, 514, 1215, 843)
    
    plot_bunches([output, *cs_maps], ghosts, halos, save_name=plots_dir / 'halos_ghosts.png')
    print(*[m.mean() for m in cs_maps], sep='\n', end='\n\n')
    print(*[(cs.mean(), l.mean(), cs.mean() * l.mean(), (cs*l).mean()) for cs, l in zip(cs_maps, l_maps)], sep='\n')
    # plot_imgs(output, cut_region_from_bunch([output]))

def transform_bunch_to_img_shape(bunch, shape):
    bunch_tensor = []
    for img in bunch:
        tensor = torch.tensor(F.interpolate(img.unsqueeze(0), (shape[2], shape[3]))) / 255
        bunch_tensor.append(tensor)
    bunch = torch.cat(bunch_tensor)
    return bunch

def test_halos():
    # output = decode_image(r"D:\windows\Documens\Diploma\results\fused\1dluts_eval\17.jpg") / 255
    output = decode_image(r"d:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_1_ep\17.jpg") / 255
    
    # output = decode_image(r"D:\windows\Documens\Diploma\results\fused\04\Gauss.jpg") / 255
    C, W, H = output.shape
    # output = F.interpolate(output.unsqueeze(0), [1280, int(H/W*1280)], mode='bilinear', align_corners=True)
    output = output.unsqueeze(0)
    bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\results\sequences\17'))
    plots_dir = Path('plots')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bunch = transform_bunch_to_img_shape(bunch, output.shape)

    output = output.to(device)
    bunch = bunch.to(device)

    l_map, cs_map, _ = mef_ssim(output, bunch, is_lum=True, full_map=True)

    output = output.squeeze()
    # ghosts = cut_region_from_bunch([output, l_map, cs_map], 1115, 1622, 1310, 2121)
    halos = cut_region_from_bunch([output, l_map, cs_map], 496, 514, 1215, 843)
    
    plot_bunches([output, l_map, cs_map], halos, save_name=plots_dir / 'halos_enhanced.png')
    print(l_map.mean(), cs_map.mean(), l_map.mean()*cs_map.mean())
    # print(*[m.mean() for m in cs_maps], sep='\n', end='\n\n')
    # print(*[(cs.mean(), l.mean(), cs.mean() * l.mean(), (cs*l).mean()) for cs, l in zip(cs_maps, l_maps)], sep='\n')
    # plot_imgs(output, cut_region_from_bunch([output]))

def test_grad():
    output = decode_image(r"d:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_1_ep\17.jpg") / 255
    
    # output = decode_image(r"D:\windows\Documens\Diploma\results\fused\04\Gauss.jpg") / 255
    C, W, H = output.shape
    # output = F.interpolate(output.unsqueeze(0), [1280, int(H/W*1280)], mode='bilinear', align_corners=True)
    output = output.unsqueeze(0)
    bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\results\sequences\17'))
    plots_dir = Path('plots')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bunch = transform_bunch_to_img_shape(bunch, output.shape)

    output = output.to(device)
    bunch = bunch.to(device)

    # calc_grad_ssim(output, bunch, 11)
    gmef_ssim(output, bunch, 11)

if __name__ == '__main__':
    from data import BracketingBunch
    from pathlib import Path
    # import numpy as np
    import cv2

    # test_mefssim()
    # test_meflut()
    # test_halos()
    test_grad()
    if False:
        output = decode_image(r"D:\windows\Documens\Diploma\results\fused\1dluts_eval\17.jpg") / 255
        # output = decode_image(r"D:\windows\Documens\Diploma\results\fused\04\Gauss.jpg") / 255
        C, W, H = output.shape
        # output = F.interpolate(output.unsqueeze(0), [1280, int(H/W*1280)], mode='bilinear', align_corners=True)
        output = output.unsqueeze(0)
        bunch = BracketingBunch(Path(r'D:\windows\Documens\Diploma\results\sequences\17'))

        bunch_tensor = []
        for img in bunch:
            tensor = torch.tensor(F.interpolate(img.unsqueeze(0), (output.shape[2], output.shape[3]))) / 255
            bunch_tensor.append(tensor)
        bunch = torch.cat(bunch_tensor)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        output = output.to(device)
        bunch = bunch.to(device)
        loss_fn = GMEF_MSSSIM()
        # for _ in range(10):
        loss = loss_fn(output, bunch)
        print(loss)