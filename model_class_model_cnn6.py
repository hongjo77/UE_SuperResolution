import torch.multiprocessing as mp
import json
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
import pytorch_lightning as pl
import lpips
import torch.optim as optim
import torch.nn as nn
import torch
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
mp.set_start_method('spawn', force=True)


class ShallowFeatures(nn.Module):
    def __init__(self, gc=64):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, gc, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(gc, gc, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv_layers(x)


class DeepFeatureBlock(nn.Module):
    def __init__(self, inp, oup, expansion=4):
        super().__init__()
        hidden_dim = int(inp * expansion)
        self.use_residual = inp == oup
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        )

        if inp != oup:
            self.residual = nn.Conv2d(inp, oup, kernel_size=1, bias=False)
        else:
            self.residual = None

    def forward(self, x):
        if self.residual is not None:
            return self.residual(x) + self.conv(x)
        else:
            return x + self.conv(x) if self.use_residual else self.conv(x)


class LightSE(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(channel, channel, kernel_size=1, bias=True)

    def forward(self, x):
        y = torch.sigmoid(self.conv(x))
        return x * y


class DynamicGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, cnn_feat1, cnn_feat2):
        importance = self.gate(cnn_feat1)
        return importance * cnn_feat1 + (1 - importance) * cnn_feat2


class FeatureExtraction(nn.Module):
    def __init__(self, dim, num_deep_blocks=1):
        super().__init__()
        self.deep_blocks1 = nn.ModuleList(
            [DeepFeatureBlock(dim, dim) for _ in range(num_deep_blocks)])
        self.lightse = LightSE(dim)

        self.deep_blocks2 = nn.ModuleList(
            [DeepFeatureBlock(dim, dim) for _ in range(num_deep_blocks)])
        self.globals = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.fusion = DynamicGate(dim)

    def forward(self, x):
        cnn_feat1 = x
        for block in self.deep_blocks1:
            cnn_feat1 = block(cnn_feat1)
        cnn_feat1 = self.lightse(cnn_feat1)

        cnn_feat2 = x
        for block in self.deep_blocks2:
            cnn_feat2 = block(cnn_feat2)
        global_map = self.globals(cnn_feat2)
        cnn_feat2 = cnn_feat2 * global_map

        fused = self.fusion(cnn_feat1, cnn_feat2)
        return fused


class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self, dim, num_deep_blocks=1, scales=[1]):
        super().__init__()
        self.extractors = nn.ModuleList([
            FeatureExtraction(dim, num_deep_blocks)
            for _ in scales
        ])
        self.channel_align = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=1, bias=False)
            for _ in scales
        ])
        self.channel_reduce = nn.Conv2d(
            dim * len(scales), dim, kernel_size=1, bias=False)

    def forward(self, x):
        features = [align(extractor(x)) for extractor, align in zip(
            self.extractors, self.channel_align)]
        features = torch.cat(features, dim=1)
        return self.channel_reduce(features)


class CustomSRModel(nn.Module):
    def __init__(self, dim=96, num_blocks=2, num_deep_blocks=2):
        super().__init__()
        self.sfeat = ShallowFeatures(gc=64)
        self.transition = nn.Conv2d(64, dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            MultiScaleFeatureExtraction(dim, num_deep_blocks)
            for _ in range(num_blocks)
        ])
        self.upscale = nn.Sequential(
            nn.Conv2d(dim, 48, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        x = self.sfeat(x)
        x = self.transition(x)
        for block in self.blocks:
            x = x + block(x)
        return self.upscale(x)


class LightningCustomSRModel(pl.LightningModule):
    def __init__(self, model, learning_rate=0.001, results_path="epoch_results.json"):
        super().__init__()
        self.model = model
        self.criterion = nn.L1Loss()
        self.learning_rate = learning_rate

        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        sys.stdout.close()
        sys.stdout = original_stdout

        self.train_losses = []
        self.val_losses = []
        self.val_psnr_scores = []
        self.val_ssim_scores = []
        self.val_lpips_scores = []
        self.results_path = results_path

        if os.path.exists(self.results_path):
            with open(self.results_path, "r") as f:
                results = json.load(f)
                self.train_losses = results.get("train_losses", [])
                self.val_losses = results.get("val_losses", [])
                self.val_psnr_scores = results.get("val_psnr_scores", [])
                self.val_ssim_scores = results.get("val_ssim_scores", [])
                self.val_lpips_scores = results.get("val_lpips_scores", [])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch
        outputs = self(lr_imgs)
        loss = self.criterion(outputs, hr_imgs)
        self.log('train_loss', loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr_imgs, hr_imgs = batch

        outputs = self(lr_imgs)

        outputs_np = outputs.detach().cpu().numpy()
        hr_imgs_np = hr_imgs.detach().cpu().numpy()

        psnr_scores = []
        ssim_scores = []

        for i in range(outputs_np.shape[0]):
            sr_img = outputs_np[i].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            hr_img = hr_imgs_np[i].transpose(1, 2, 0)  # (C, H, W, C)

            if sr_img.shape[0] < 7 or sr_img.shape[1] < 7:
                continue

            psnr_scores.append(peak_signal_noise_ratio(
                sr_img, hr_img, data_range=1))

            ssim_scores.append(structural_similarity(
                sr_img, hr_img, channel_axis=2, win_size=7, data_range=1))

        psnr = np.mean(psnr_scores) if psnr_scores else 0
        ssim = np.mean(ssim_scores) if ssim_scores else 0

        lpips_score = self.loss_fn_alex(outputs, hr_imgs).mean()

        self.log('val_loss', self.criterion(outputs, hr_imgs),
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_psnr', psnr, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ssim', ssim, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_lpips', lpips_score, on_epoch=True,
                 prog_bar=True, logger=True)

        return {'val_psnr': psnr, 'val_ssim': ssim, 'val_lpips': lpips_score}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics['train_loss_epoch'].item()
        self.train_losses.append(train_loss)
        print(f"Epoch {self.current_epoch}: Train Loss: {train_loss:.4f}")

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics['val_loss'].item()
        val_psnr = self.trainer.callback_metrics['val_psnr'].item()
        val_ssim = self.trainer.callback_metrics['val_ssim'].item()
        val_lpips = self.trainer.callback_metrics['val_lpips'].item()

        self.val_losses.append(val_loss)
        self.val_psnr_scores.append(val_psnr)
        self.val_ssim_scores.append(val_ssim)
        self.val_lpips_scores.append(val_lpips)

        print(f"Epoch {self.current_epoch}: Validation Loss: {val_loss:.4f}, "
              f"PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}, LPIPS: {val_lpips:.4f}")

        results = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_psnr_scores": self.val_psnr_scores,
            "val_ssim_scores": self.val_ssim_scores,
            "val_lpips_scores": self.val_lpips_scores,
        }
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Epoch results appended to '{self.results_path}'.")
