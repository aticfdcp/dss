from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model_path=None, confidence=0.20):
        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = (
            Path(model_path)
            if model_path
            else self.project_root / "models" / "yolo11n.pt"
        )

        self.confidence = confidence
        self.device = 0 if torch.cuda.is_available() else "cpu"

        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")

        self.model = YOLO(str(self.model_path))

        print(f"Modelo: {self.model_path}")
        print(f"Dispositivo: {'GPU' if self.device == 0 else 'CPU'}")

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence,
            imgsz=960,
            device=self.device,
            verbose=False,
        )

        return results[0]


if __name__ == "__main__":
    detector = PersonDetector()

    video_path = detector.project_root / "recordings" / "input" / "test.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    output_dir = detector.project_root / "recordings" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test_detection.mp4"

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = 2560
    height = 1440

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el video de salida.")

    frame_count = 0
    max_frames = 300

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            #frame = cv2.resize(frame, (width, height))

            result = detector.detect(frame)

            annotated_frame = result.plot()
            writer.write(annotated_frame)

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Fotogramas procesados: {frame_count}")

    finally:
        cap.release()
        writer.release()

    print("\nPrueba finalizada.")
    print(f"Fotogramas procesados: {frame_count}")
    print(f"Video generado: {output_path}")
