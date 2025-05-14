import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

import os
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from losses.gssim import GSSIM
from losses.mef_ssim import MEF_MSSSIM
from model.efficient_hdr_light import EfficientHDR
import math

import gc

def save_image_tensor(tensor: torch.Tensor, name: Path):
    img = tensor.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def bunch_to_Y_CbfCrf(bunch):
    Y = bunch[:, 0, :, :].unsqueeze(1)
    Cb = bunch[:, 1, :, :].unsqueeze(1)
    Cr = bunch[:, 2, :, :].unsqueeze(1)

    Wb = torch.abs(Cb - 0.5) / torch.sum(torch.abs(Cb - 0.5).clamp(1e-6), dim=0)
    Wr = torch.abs(Cr - 0.5) / torch.sum(torch.abs(Cr - 0.5).clamp(1e-6), dim=0)
    Cb_f = torch.sum(Wb * Cb, dim=0, keepdim=True).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr, dim=0, keepdim=True).clamp(0, 1)
    return Y, Cb_f, Cr_f

def fuse_YCbCr(bunch: torch.Tensor, weights: torch.tensor, masks: torch.Tensor):
    Y = bunch[:, 0].unsqueeze(1)
    Cb = bunch[:, 1].unsqueeze(1)
    Cr = bunch[:, 2].unsqueeze(1)
    
    Wy = (Y * weights * masks) / torch.sum(Y * weights * masks, dim=0).clamp(1e-6)
    Wb = (torch.abs(Cb - 0.5) * masks) / torch.sum(torch.abs(Cb - 0.5)*masks, dim=0).clamp(1e-6)
    Wr = (torch.abs(Cr - 0.5) * masks) / torch.sum(torch.abs(Cr - 0.5)*masks, dim=0).clamp(1e-6)

    Y_f = torch.sum(Wy * Y, dim=0, keepdim=False).clamp(0, 1)
    Cb_f = torch.sum(Wb * Cb, dim=0, keepdim=False).clamp(0, 1)
    Cr_f = torch.sum(Wr * Cr, dim=0, keepdim=False).clamp(0, 1)
    
    return Y_f, Cb_f, Cr_f

def fuse_YCbCr_(bunch: torch.Tensor, weights: torch.Tensor, masks: torch.Tensor, eps=1e-6):
    """
    bunch: [K, 3, H, W] - тензор в пространстве YCbCr
    weights: [K, 1, H, W] - исходные веса
    masks: [K, 1, H, W] - бинарные маски (1 = валидная область)
    eps: стабилизирующая константа
    """
    # 1. Расширение масок для плавных переходов
    kernel = torch.ones(1, 1, 3, 3, device=masks.device)
    dilated_masks = 1 - F.conv2d(1 - masks.float(), kernel, padding=1).clamp(0, 1)
    
    # 2. Комбинирование весов с масками
    combined_weights = weights * dilated_masks
    
    # 3. Разделение каналов
    Y = bunch[:, 0].unsqueeze(1)
    Cb = bunch[:, 1].unsqueeze(1)
    Cr = bunch[:, 2].unsqueeze(1)
    
    # 4. Веса для яркостного канала (Y)
    Wy = combined_weights / (combined_weights.sum(dim=0) + eps)
    
    # 5. Веса для цветовых каналов (Cb/Cr) с учётом цветовой согласованности
    W_color = (dilated_masks * torch.exp(-10*(Cb - 0.5).abs()) * 
              torch.exp(-10*(Cr - 0.5).abs()))
    W_color = W_color / (W_color.sum(dim=0) + eps)
    
    # 6. Взвешенное усреднение
    Y_fused = (Y * Wy).sum(dim=0)
    Cb_fused = (Cb * W_color).sum(dim=0)
    Cr_fused = (Cr * W_color).sum(dim=0)
    
    return Y_fused.clamp(0,1), Cb_fused.clamp(0,1), Cr_fused.clamp(0,1)

import torch
import torch.nn.functional as F

def ghost_free_fusion(images, weights, occlusion_masks, dilation_kernel_size=3):
    """
    images: [K, 3, H, W] - тензор кадров брекетинга
    weights: [K, 1, H, W] - исходные веса без учета движения
    occlusion_masks: [K, 1, H, W] - маски окклюзий (1 - движущийся объект)
    dilation_kernel_size: размер ядра для расширения масок
    """
    K, _, H, W = images.shape
    
    # 1. Расширение масок движущихся объектов
    kernel = torch.ones((1, 1, dilation_kernel_size, dilation_kernel_size), 
                      device=occlusion_masks.device)
    expanded_masks = []
    for k in range(K):
        mask = occlusion_masks[k]  # [1, H, W]
        # Применяем дилатацию через свертку
        dilated = F.conv2d(mask.float(), kernel, padding=dilation_kernel_size//2)
        dilated = (dilated > 0).float()
        expanded_masks.append(dilated)
    expanded_masks = torch.stack(expanded_masks)  # [K, 1, H, W]

    # 2. Обнуление весов в расширенных масках
    modified_weights = weights * (1 - expanded_masks)

    # 3. Нормализация весов (сумма по K = 1 для каждого пикселя)
    weight_sum = modified_weights.sum(dim=0, keepdim=True) + 1e-8
    normalized_weights = modified_weights / weight_sum

    # 4. Взвешенное суммирование кадров
    fused = (images * normalized_weights).sum(dim=0)
    
    return fused.clamp(0, 1)

class EarlyStopper:
    def __init__(self, model: nn.Module, weights_name, scripted_name, save_dir, patience=10, delta=3):
        self.model = model
        os.makedirs(save_dir, exist_ok=True)
        self.weights_name = save_dir / weights_name
        self.scripted_name = save_dir / scripted_name

        self.patience = patience
        self.delta = delta

        self.counter = 0

        self.best_loss = float('inf')
        self.best_f1 = 0.0


    def __call__(self, valid_loss, valid_f1=0.0):
        match self.early_stop(valid_loss, valid_f1):
            case 0:
                print(f'\nEarly stopped: best_loss = {self.best_loss:.3} | best_f1 = {self.best_f1:.3}')
                print(f'Model weights saved with name  "{self.weights_name}"')
                print(f'Scripted model saved with name "{self.scripted_name}"')
                return True
            case 2:
                self.save_model()
                return False
            case _:
                return False


    def save_model(self):
        torch.save(self.model, self.weights_name)
        try:
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(self.scripted_name)
        except RuntimeError: # конвертированные модели не скриптуются
            pass


    def early_stop(self, valid_loss, valid_f1):
        """
        0: stop
        1: continue
        2: save best
        """
        valid_loss = float(f'{valid_loss:.{self.delta}}')
        valid_f1 = float(f'{valid_f1:.{self.delta}}')
        if valid_f1 > self.best_f1:
            self.best_f1 = valid_f1
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
            self.counter = 0
            return 2
        elif valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.counter = 0
            return 2
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return 0

        return 1
    

class Trainer():
    '''
    Classification task trainer. Should be reimplemented through a base class
    '''
    def __init__(self, model: nn.Module, train_loader, valid_loader, test_loader, name: str, save_ckpt_dir: Path, save_res_dir: Path, weights=None, transform_out=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.weights_name = f'{name}_weights.pt'
        self.scripted_name = f'{name}_scripted.pt'

        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader

        if weights is not None:
          weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # self.class_criterion = nn.CrossEntropyLoss(weight=weights)

        self.metric = MEF_MSSSIM(11)
        
        self.regression_criterion = nn.MSELoss()

        self.train_losses = []
        self.train_f1s = []

        self.valid_losses = []
        self.valid_f1s = []

        self.early_stopper = EarlyStopper(
                self.model,
                self.weights_name,
                self.scripted_name,
                save_ckpt_dir,
                patience=2,
                delta=3,
        )

        self.save_dir = Path(save_res_dir)
        self.train_results = self.save_dir / 'train'
        self.train_results_images = self.train_results / 'images'
        self.train_results_weights = self.train_results / 'weights'

        self.test_results = self.save_dir / 'test'
        self.test_results_images = self.test_results / 'images'
        self.test_results_weights = self.test_results / 'weights'

        for dir in [
            self.train_results_images, self.train_results_weights,
            self.test_results_images, self.test_results_weights
        ]:
            os.makedirs(dir, exist_ok=True)

        self.transform_out = transform_out

        torch.autograd.set_detect_anomaly(True)


    def get_best_model(self):
        if os.path.exists(self.scripted_name):
            print(f'loaded best from disk {self.scripted_name}')
            return torch.jit.load(self.scripted_name)
        return self.model


    def train(self, epoch_num, lr=0.0005, load_from_disk=False, epochs_per_val=5, save_val=False):
        gc.collect()
        torch.cuda.empty_cache()
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                # betas=(0.999, 0.9999),
                weight_decay=0.01,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=1e-5, T_max=epoch_num)
        self.early_stopper.counter = 0

        if load_from_disk:
            self.model = self.get_best_model()
        # else:
        #     if os.path.exists(self.weights_name): os.remove(self.weights_name)
        #     if os.path.exists(self.scripted_name): os.remove(self.scripted_name)

        self.model.to(self.device)
        for epoch in range(1, epoch_num+1):
            print(f'\nEpoch: {epoch} / {epoch_num}')

            train_loss = self.train_step(self.train_loader, save=save_val)
            self.train_losses.append(train_loss)

            self.scheduler.step()

            if epoch % epochs_per_val == 0:
                valid_loss = self.valid_step(self.valid_loader, save=save_val)
                self.valid_losses.append(valid_loss)
                
                if self.early_stopper(valid_loss):
                    break
                
        gc.collect()
        torch.cuda.empty_cache()


    def report(self, max_classes_not_to_reduce=20):
        print('\nTest-Epoch')
        self.model = self.get_best_model()
        self.model.to(self.device)
        confmat = self.valid_step(self.valid_loader, return_confmat=True)

        report = confmat.classification_report(output_dict=False)
        if confmat.cls_num > max_classes_not_to_reduce:
            print(f'\n\nlen(classes) is bigger than {max_classes_not_to_reduce}, so output will be reduced')
            report_list = report.split('\n')
            report = '\n'.join([*report_list[:2], *report_list[-3:]])
        print('\n', report)
        return confmat


    def train_step(self, train_loader, save=False, transform_out=None) -> float:
        self.model.train()
        run_loss = 0.0

        for name, exposures, corrected, weights in (pbar := tqdm(train_loader)):
            batch_size = len(exposures)
            B, K, C, H, W = exposures.shape
            exposures = exposures.to(self.device).squeeze(0)
            corrected = corrected.to(self.device).squeeze(0)
            weights = weights.to(self.device).squeeze(0)
            
            masks = self.model(corrected)
            Y, Cb, Cr = fuse_YCbCr(exposures, weights, masks)

            loss = -self.metric(Y.unsqueeze(0), corrected)
            run_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            if save:
                output = torch.cat((Y, Cb, Cr), dim=0).unsqueeze(0)
                self.save_results(output, masks, name, is_train=True)

            pbar.set_postfix({'gssim ': f' {-loss:.3}',})

            torch.cuda.empty_cache()
            # break

        train_loss = run_loss / len(train_loader)
        print(f'  train gssim:\t {-train_loss:.3}')
        

        return train_loss


    def valid_step(self, valid_loader, save=False, transform_out=None) -> float:
        # self.model.eval()
        # https://github.com/microsoft/Swin3D/issues/17
        self.model.eval()
        run_loss = 0.0

        with torch.no_grad():
            for name, exposures, corrected, weights in (pbar := tqdm(valid_loader)):
                batch_size = len(exposures)
                B, K, C, H, W = exposures.shape
                exposures = exposures.to(self.device).squeeze(0)
                corrected = corrected.to(self.device).squeeze(0)
                weights = weights.to(self.device).squeeze(0)
                
                masks = self.model(corrected)
                Y, Cb, Cr = fuse_YCbCr(exposures, weights, masks)

                loss = -self.metric(Y.unsqueeze(0), corrected)
                run_loss += loss.item()

                if save:
                    output = torch.cat((Y, Cb, Cr), dim=0).unsqueeze(0)
                    self.save_results(output, masks, name, is_train=False)

                pbar.set_postfix({'gssim ': f' {-loss:.3}',})
                
                torch.cuda.empty_cache()
                # break

        valid_loss = run_loss / len(valid_loader)
        print(f'  valid gssim:\t {-valid_loss:.3}')

        return valid_loss
    
    def save_results(self, output: torch.Tensor, weights: torch.Tensor, name, is_train=True):
        if is_train:
            images_dir = self.train_results_images
            weights_dir = self.train_results_weights
        else:
            images_dir = self.test_results_images
            weights_dir = self.test_results_weights
        
        name = name[0]
        weights_dir = weights_dir / name
        os.makedirs(weights_dir, exist_ok=True)

        if self.transform_out:
            output = self.transform_out(output) 

        save_image_tensor(output, (images_dir / name).with_suffix('.jpg'))
        for i, weight in enumerate(weights):
            save_image_tensor(weight.expand_as(output), (weights_dir / str(i)).with_suffix('.jpg'))
    
    
    def predict(self, valid_loader) -> np.ndarray:
        self.model.eval()
        preds = []

        for bunch in tqdm(valid_loader):
            bunch = bunch.to(self.device)
            outputs = self.model(bunch).detach().numpy()
            pred = np.argmax(outputs, axis=1 if len(outputs.shape) > 1 else 0)
            preds = [*preds, *pred]

        return np.array(preds)
    

    def plot(self):
        epochs = list(range(1, len(self.train_losses)+1))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

        axes[0].plot(epochs, -self.train_losses, '--b',label='train')
        axes[0].plot(epochs, -self.valid_losses, 'r',label='valid')
        axes[0].set(xlabel='epoch num', ylabel='GMEF_SSIM', title='GMEF_SSIM')
        axes[0].grid()
        axes[0].legend()

        # axes[1].plot(epochs, self.train_f1s, '--b', label='train')
        # axes[1].plot(epochs, self.valid_f1s, 'r', label='valid')
        # axes[1].set(xlabel='epoch num', ylabel='f1', title='F1-score')
        # axes[1].grid()
        # axes[1].legend()

        fig.tight_layout()
        plt.show() 


if __name__ == '__main__':
    from data import BunchWeightedDataset, RGBToYCbCr, YCbCrToRGB
    import torchvision.transforms.v2 as v2
    
    transform_exp = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((1248, 1662)),
        RGBToYCbCr(),
    ])

    transform_w = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Resize((1248, 1662)),
    ])

    transform_out = YCbCrToRGB()
    
    train_set = BunchWeightedDataset(
        r'D:\windows\Documens\Diploma\dataset\train\train', 
        r'D:\windows\Documens\Diploma\results\fused\3_1_scale_mef_ssim_lum_0_ep\train\weights',
        num_exposures=3, 
        transform_exposure=transform_exp,
        transform_w=transform_w,
    )

    _, exps, corrected, weights =  train_set[0]
    save_image_tensor()