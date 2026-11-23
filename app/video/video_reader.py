import cv2
import os


class VideoReader:

    def __init__(self, video_path):

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No existe el archivo: {video_path}")

        self.video_path = video_path

        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise RuntimeError("No se pudo abrir el video.")

        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.duration = self.total_frames / self.fps if self.fps > 0 else 0

    def get_information(self):

        return {
            "filename": os.path.basename(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "total_frames": self.total_frames,
            "duration_seconds": round(self.duration, 2),
        }

    def read_frame(self):

        success, frame = self.capture.read()

        if not success:
            return None

        return frame

    def release(self):

        if self.capture is not None:
            self.capture.release()
