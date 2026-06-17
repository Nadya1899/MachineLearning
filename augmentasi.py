import os
import cv2
import numpy as np
from tqdm import tqdm

class ImageAugmenter:
    def __init__(self, input_dir, output_dir, multiplier=5):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.multiplier = multiplier

    def augment_images(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(".jpg") or f.endswith(".png")]

        for img_file in tqdm(image_files, desc="Augmenting Images"):
            img_path = os.path.join(self.input_dir, img_file)
            image = cv2.imread(img_path)

            for i in range(1, self.multiplier + 1):
                augmented_img = cv2.flip(image, flipCode=i)  # Augmentasi dengan flip
                augmented_img_path = os.path.join(self.output_dir, f"{os.path.splitext(img_file)[0]}_{i}.jpg")
                cv2.imwrite(augmented_img_path, augmented_img)

if __name__ == "__main__":
    input_dir = "D:\SEMESTER 4\PEMBELAJARAN MESIN\Deteksi_NovelOri\Buku Bajakan"
    output_dir = "D:\SEMESTER 4\PEMBELAJARAN MESIN\Deteksi_NovelOri\Buku Bajakan"
    multiplier = 5  # Faktor untuk memperbanyak citra

    augmenter = ImageAugmenter(input_dir, output_dir, multiplier)
    augmenter.augment_images()
