from inference import ImageDenoiser

def main():
    denoiser = ImageDenoiser(weights_path="models/best_weight.pth")
    images_to_process = ["test_images/01.jpg", "test_images/02.png"]
    for img_path in images_to_process:
        try:
            orig_pil, clean_pil = denoiser.process_image(img_path)
            saved_path = denoiser.save_result(img_path, clean_pil)
            print(f"Успешно обработано: {saved_path}")
        except Exception as e:
            print(f"Ошибка с файлом {img_path}: {e}")

if __name__ == "__main__":
    main()
