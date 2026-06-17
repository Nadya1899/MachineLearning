import cv2
import os

class VideoFrameExtractor:
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        self.output_folder = "output_frames"  # Nama folder untuk menyimpan frame

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not self.cap.isOpened():
            print("Gagal membuka video")
            exit()

    def extract_frames(self):
        frame_number = 1
        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("Selesai membaca video")
                break

            # Simpan frame ke dalam folder
            frame_name = os.path.join(self.output_folder, f"frame_{frame_number}.jpg")
            cv2.imwrite(frame_name, frame)

            frame_number += 1

    def release(self):
        self.cap.release()

# Contoh penggunaan kelas VideoFrameExtractor
if __name__ == "__main__":
    video_file = 'D:\SEMESTER 4\PEMBELAJARAN MESIN\DCIM\video_20240501_193031.mp4'  # Ganti dengan path video Anda
    extractor = VideoFrameExtractor(video_file)
    extractor.extract_frames()
    extractor.release()
