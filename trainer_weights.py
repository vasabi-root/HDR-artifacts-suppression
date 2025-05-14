import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from losses.gmef_ssim import GMEF_MSSSIM
from model.efficient_hdr_light import EfficientHDR
import math

import gc

def save_image_tensor(tensor: torch.Tensor, name: Path):
    img = tensor.squeeze(0).mul(255).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    cv2.imwrite(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

class ConfusionMatrix():
    def __init__(self, labels):
        self.labels = np.array(labels) # idx 0: nocall
        self.cls_num = len(self.labels)
        # row == True : col == Pred
        self.matrix = np.array([[0]*self.cls_num for _ in range(self.cls_num)])

        self.true = []
        self.pred = []


    def __call__(self, output, target):
        return self.calc_confmat(output, target)


    def calc_confmat(self, outputs, targets):
        # multilabel classification:
        # outputs.shape = [Batch_N, N]
        # targets.shape = [Batch_N, N]
        if torch.is_tensor(outputs) or torch.is_tensor(targets):
            outputs = outputs.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()

        threshold = 0.5
        output_classes = (outputs > threshold)
        target_classes = (targets > threshold)

        # self.matrix[target_classes, output_classes] += 1
        
        # for further multilabel 'classification_report()'
        self.true = [*self.true, *target_classes]
        self.pred = [*self.pred, *output_classes]

    def f1_score(self):
        return f1_score(self.true, self.pred, average='weighted', zero_division=0.0,)
        
    def classification_report(self, output_dict=False):
        cls_report = classification_report(
            self.true,
            self.pred,
            target_names=self.labels,
            output_dict=output_dict,
            zero_division=0.0,
        )
        return cls_report


    def plot(self):
        figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(self.matrix, display_labels=self.labels)
        disp.plot(ax=ax)
        plt.show()


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
    def __init__(self, model: nn.Module, train_loader, valid_loader, test_loader, name: str, save_ckpt_dir: Path, save_res_dir: Path, weights=None):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.model = model.to(self.device)

        self.weights_name = f'{name}_weights.pt'
        self.scripted_name = f'{name}_scripted.pt'

        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader

        if weights is not None:
          weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        # self.class_criterion = nn.CrossEntropyLoss(weight=weights)

        self.metric = GMEF_MSSSIM(11)
        
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
                patience=5,
                delta=3,
        )

        self.save_dir = Path(save_res_dir)
        os.makedirs(self.save_dir, exist_ok=True)


    def get_best_model(self):
        if os.path.exists(self.scripted_name):
            print(f'loaded best from disk {self.scripted_name}')
            return torch.jit.load(self.scripted_name)
        return self.model


    def train(self, epoch_num, lr=0.001, load_from_disk=False, epochs_per_val=5, save_val=False, transform_out=None):
        gc.collect()
        torch.cuda.empty_cache()
        
        self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                # betas=(0.999, 0.9999),
                # weight_decay=0.01,
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

            train_loss = self.train_step(self.train_loader, save=save_val, transform_out=transform_out)
            self.train_losses.append(train_loss)

            self.scheduler.step()

            if epoch % epochs_per_val == 0:
                valid_loss = self.valid_step(self.valid_loader, save=save_val, transform_out=transform_out)
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

        for name, bunch in (pbar := tqdm(train_loader)):
            self.optimizer.zero_grad()
            batch_size = len(bunch)
            # try:
            bunch = bunch.to(self.device)
            weights = self.model(bunch)
            output = (bunch * weights).sum(1) / weights.sum(1)

            loss = 1-self.metric(output, torch.squeeze(bunch, dim=0))
            # except ValueError:
            #     continue

            loss.backward()
            self.optimizer.step()

            if save:
                if transform_out:
                    output = transform_out(output)
                save_image_tensor(output, (self.save_dir / name[0]).with_suffix('.jpg'))

            run_loss += loss.item()
            pbar.set_postfix({'gmef_ssim ': f' {1-loss:.3}',})

            torch.cuda.empty_cache()
            # break

        train_loss = run_loss / len(train_loader)
        print(f'  train gmef_ssim:\t {1-train_loss:.3}')
        

        return train_loss


    def valid_step(self, valid_loader, save=False, transform_out=None) -> float:
        # self.model.eval()
        # https://github.com/microsoft/Swin3D/issues/17
        self.model.eval()
        run_loss = 0.0

        with torch.no_grad():
            for name, bunch in (pbar := tqdm(valid_loader)):
                batch_size = len(bunch)
                # try:
                bunch = bunch.to(self.device)
                weights = self.model(bunch)
                output = (bunch * weights).sum(1) / weights.sum(1)
    
                loss = 1-self.metric(output, torch.squeeze(bunch, dim=0))
                # except ValueError:
                #     continue
                
                run_loss += loss.item()

                if save:
                    if transform_out:
                        output = transform_out(output)
                    save_image_tensor(output, (self.save_dir / name[0]).with_suffix('.jpg'))

                pbar.set_postfix({'gmef_ssim ': f' {1-loss:.3}',})
                
                torch.cuda.empty_cache()
                # break

        valid_loss = run_loss / len(valid_loader)
        print(f'  valid gmef_ssim:\t {1-valid_loss:.3}')

        return valid_loss
    
    
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

        axes[0].plot(epochs, 1-self.train_losses, '--b',label='train')
        axes[0].plot(epochs, 1-self.valid_losses, 'r',label='valid')
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
