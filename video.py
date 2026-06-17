import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class VideoProcessor:
    def __init__(self, video_file):
        self.video_file = video_file
        self.cap = cv2.VideoCapture(video_file)
        self.output_folder = "output_frames1"  # Nama folder untuk menyimpan frame

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        if not self.cap.isOpened():
            print("Gagal membuka video")
            exit()

    def process_frame(self, frame):
        # Contoh: ubah frame menjadi grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray_frame

    def save_frame(self, frame, frame_number):
        frame_name = os.path.join(self.output_folder, f"frame_{frame_number}.png")
        cv2.imwrite(frame_name, frame)

    def start_processing(self):
        plt.figure()  # Buat satu figure di luar loop
        frame_number = 1
        while True:
            ret1, frame1 = self.cap.read()

            if not ret1:
                print("Selesai membaca video")
                break

            if frame_number == 1:
                processed_frame1 = self.process_frame(frame1)
            else:
                processed_frame2 = self.process_frame(frame1)
                # Cari absolute dari perbedaan antara frame 1 dan frame 2
                diff_frame = np.square(np.subtract(processed_frame1.astype(np.int32), processed_frame2.astype(np.int32)))
                _, thresholded_diff = cv2.threshold(diff_frame.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)

                # Lakukan operasi morfologi untuk mengisi area putih dan lubang di dalamnya
                kernel = np.ones((3,3), np.uint8)
                thresholded_diff = cv2.erode(thresholded_diff, kernel, iterations=1) 
                kernel = np.ones((5,5), np.uint8)
                filled_diff = cv2.dilate(thresholded_diff, kernel, iterations=4)

                # Simpan frame hasil proses
                self.save_frame(filled_diff, frame_number)

                processed_frame1 = processed_frame2

            frame_number += 1

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    def release(self):
        self.cap.release()

# Contoh penggunaan kelas VideoProcessor
if __name__ == "__main__":
    video_file = 'D:\SEMESTER 4\PEMBELAJARAN MESIN\code_video\ido_walk.avi'
    processor = VideoProcessor(video_file)
    processor.start_processing()
    processor.release()
