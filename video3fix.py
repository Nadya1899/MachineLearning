import cv2
import os
import glob

class VideoFrameExtractor:
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.output_folder = "output_frames3"  # Nama folder untuk menyimpan frame

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def extract_frames(self):
        video_files = glob.glob(os.path.join(self.video_folder, '*.mp4'))

        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            frame_number = 1

            while True:
                ret, frame = cap.read()

                if not ret:
                    print(f"Selesai membaca {video_file}")
                    break

                # Simpan frame ke dalam folder
                frame_name = os.path.join(self.output_folder, f"{os.path.basename(video_file)}_frame_{frame_number}.jpg")
                cv2.imwrite(frame_name, frame)

                frame_number += 1

            cap.release()

    def release(self):
        pass  # Tidak perlu melakukan release

# Contoh penggunaan kelas VideoFrameExtractor
if __name__ == "__main__":
    video_folder = 'D:\SEMESTER 4\PEMBELAJARAN MESIN\DCIM\Ori'  
    extractor = VideoFrameExtractor(video_folder)
    extractor.extract_frames()
