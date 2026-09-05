import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.amp import GradScaler, autocast
import math

from models import UNetDenoiser, SSIMLoss, PerceptualLoss
from utils.dataset import DevisingDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используем устройство: {device}")

    BASE_DIR = r"D:\Project ai\processed"
    SIDD_DIR = os.path.join(BASE_DIR, "sidd_real")
    DIV2K_DIR = os.path.join(BASE_DIR, "synthetic_div2k")

    print("Загрузка датасетов")
    train_dataset = ConcatDataset([
        DevisingDataset(root_dir=SIDD_DIR, split='train'),
        DevisingDataset(root_dir=DIV2K_DIR, split='train')
    ])

    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = ConcatDataset([
        DevisingDataset(root_dir=SIDD_DIR, split='val'),
        DevisingDataset(root_dir=DIV2K_DIR, split='val')
    ])
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetDenoiser().to(device)

    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIMLoss()
    vgg_criterion = PerceptualLoss(device)

    start_epoch = 70
    total_epochs = 80
    checkpoint_path = "my_denoiser_up_epoch_70.pth"

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Веса модели успешно загружены!")
    else:
        print(f"Ошибка: Файл {checkpoint_path} не найден в папке проекта!")
        return

    optimizer = optim.Adam(model.parameters(), lr=5e-5)

    epochs_left = total_epochs - start_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_left, eta_min=5e-6)

    scaler = GradScaler('cuda')

    print(f"Начальный LR: {optimizer.param_groups[0]['lr']:.6f}\n")

    for epoch in range(start_epoch, total_epochs):
        torch.cuda.empty_cache()
        model.train()
        running_train_loss = 0.0

        for i, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs = noisy_imgs.to(device, non_blocking=True)
            clean_imgs = clean_imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                outputs = model(noisy_imgs)
                outputs_clamped = torch.clamp(outputs, 0.0, 1.0)

                l1_loss = l1_criterion(outputs_clamped, clean_imgs)
                ssim_loss = ssim_criterion(outputs_clamped, clean_imgs)
                vgg_loss = vgg_criterion(outputs_clamped, clean_imgs)
                loss = 0.3 * l1_loss + 0.7 * ssim_loss + 0.25 * vgg_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            if (i + 1) % 200 == 0:
                print(
                    f"Эпоха [{epoch + 1}/{total_epochs}], Шаг [{i + 1}/{len(train_loader)}], Train Loss: {running_train_loss / 200:.5f}")
                running_train_loss = 0.0

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for noisy_imgs, clean_imgs in val_loader:
                noisy_imgs = noisy_imgs.to(device, non_blocking=True)
                clean_imgs = clean_imgs.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(noisy_imgs)
                    outputs_clamped = torch.clamp(outputs, 0.0, 1.0)

                    l1_val = l1_criterion(outputs_clamped, clean_imgs)
                    ssim_val = ssim_criterion(outputs_clamped, clean_imgs)
                    vgg_val = vgg_criterion(outputs_clamped, clean_imgs)

                    val_loss = 0.3 * l1_val + 0.7 * ssim_val + 0.25 * vgg_val

                running_val_loss += val_loss.item()

        total_val_loss = running_val_loss / len(val_loader)
        print(f"ИТОГ ЭПОХИ {epoch + 1}: LR = {current_lr:.6f} | Средний Val Loss = {total_val_loss:.5f}")

        save_path = f"my_denoiser_up_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Модель сохранена in {save_path}\n")


if __name__ == '__main__':
    main()
