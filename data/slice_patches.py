"""
slice_patches.py - нарезка изображений на патчи для обучения.

Нарезает пары clean/noisy изображений на перекрывающиеся патчи 256x256 с шагом 128.
Отбрасывает патчи с низким стандартным отклонением (σ < 15) — это однородные области.

Использование:
    python slice_patches.py --clean_dir /path/to/clean --noisy_dir /path/to/noisy --output_root /path/to/processed --start_idx 0 --num_images 100
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path


def imread_unicode(path):
    """Читает изображение по пути с поддержкой Unicode."""
    try:
        with open(path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Ошибка при чтении {path}: {e}")
        return None


def imwrite_unicode(path, img):
    """Сохраняет изображение в PNG с поддержкой Unicode."""
    try:
        result, nparr = cv2.imencode('.png', img)
        if result:
            nparr.tofile(path)
            return True
    except Exception as e:
        print(f"Ошибка при сохранении {path}: {e}")
        return False


def slice_patches(clean_path, noisy_path, output_root, idx, patch_size=256, stride=128, sigma_thresh=15):
    """
    Нарезает одно изображение на патчи и сохраняет в структуру папок.

    Args:
        clean_path: путь к чистому изображению
        noisy_path: путь к зашумлённому изображению (или такое же имя, если шум уже есть)
        output_root: корневая папка для сохранения (создаются clean/ и noisy/)
        idx: индекс изображения (для именования файлов)
        patch_size: размер патча (по умолчанию 256)
        stride: шаг скользящего окна (по умолчанию 128)
        sigma_thresh: порог стандартного отклонения (по умолчанию 15)
    Returns:
        количество созданных патчей
    """
    clean_dir = os.path.join(output_root, "clean")
    noisy_dir = os.path.join(output_root, "noisy")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(noisy_dir, exist_ok=True)

    img_c = imread_unicode(clean_path)
    img_n = imread_unicode(noisy_path)

    if img_c is None or img_n is None:
        return 0

    h, w = img_c.shape[:2]
    h_n, w_n = img_n.shape[:2]
    h, w = min(h, h_n), min(w, w_n)

    count = 0

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_c = img_c[y:y + patch_size, x:x + patch_size]

            if np.std(patch_c) > sigma_thresh:
                patch_n = img_n[y:y + patch_size, x:x + patch_size]

                filename = f"picture{idx:03d}_patch_{count:05d}.png"

                imwrite_unicode(os.path.join(clean_dir, filename), patch_c)
                imwrite_unicode(os.path.join(noisy_dir, filename), patch_n)

                count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Нарезка изображений на патчи для обучения.")
    parser.add_argument('--clean_dir', type=str, required=True, help='Папка с чистыми изображениями')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Папка с зашумлёнными изображениями')
    parser.add_argument('--output_root', type=str, required=True, help='Корневая папка для сохранения патчей')
    parser.add_argument('--start_idx', type=int, default=0, help='Начальный индекс для нумерации изображений')
    parser.add_argument('--num_images', type=int, default=None, help='Количество изображений для обработки (если None — все)')
    parser.add_argument('--patch_size', type=int, default=256, help='Размер патча (по умолчанию 256)')
    parser.add_argument('--stride', type=int, default=128, help='Шаг скользящего окна (по умолчанию 128)')
    parser.add_argument('--sigma_thresh', type=float, default=15.0, help='Порог стандартного отклонения (по умолчанию 15)')

    args = parser.parse_args()

    clean_files = sorted([f for f in os.listdir(args.clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    if args.num_images is not None:
        clean_files = clean_files[:args.num_images]

    total_patches = 0

    for i, filename in enumerate(clean_files):
        idx = args.start_idx + i
        clean_path = os.path.join(args.clean_dir, filename)
        noisy_path = os.path.join(args.noisy_dir, filename)

        print(f"Обработка изображения {idx}: {filename}...")
        count = slice_patches(
            clean_path, noisy_path, args.output_root, idx,
            patch_size=args.patch_size, stride=args.stride, sigma_thresh=args.sigma_thresh
        )
        total_patches += count
        print(f"  Создано патчей: {count}")

    print(f"\nВсего создано патчей: {total_patches}")


if __name__ == '__main__':
    main()
