from pathlib import Path
import cv2
import torch
import csv
import math
from ultralytics import YOLO


class PersonTracker:

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = self.project_root / "models" / "yolo11n.pt"

        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")

        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))

        print("Modelo YOLO11n cargado")
        print(f"Dispositivo: {'GPU' if self.device == 0 else 'CPU'}")

    def track(self, frame):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.30,
            iou=0.50,
            imgsz=960,
            device=self.device,
            verbose=False,
        )

        return results[0]


def main():

    tracker = PersonTracker()

    video_path = tracker.project_root / "recordings" / "input" / "test.mp4"

    output_dir = tracker.project_root / "recordings" / "processed"

    log_dir = tracker.project_root / "logs" / "tracking"

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "test_tracking.mp4"
    csv_path = log_dir / "test_tracking.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("El video tiene una frecuencia FPS inválida.")

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el video de salida.")

    max_frames = 300
    frame_count = 0

    # Historial de posiciones y velocidades por persona
    history = {}

    # Conversión temporal entre fotogramas
    frame_time = 1 / fps

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")

        csv_writer.writerow(
            [
                "fotograma",
                "tiempo_segundos",
                "id_persona",
                "centro_x",
                "centro_y",
                "ancho",
                "alto",
                "confianza",
                "velocidad_px_seg",
                "aceleracion_px_seg2",
            ]
        )

        try:

            while frame_count < max_frames:

                ret, frame = cap.read()

                if not ret:
                    break

                result = tracker.track(frame)

                annotated_frame = result.plot()

                if result.boxes is not None and result.boxes.id is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()
                    confidences = result.boxes.conf.cpu().tolist()

                    for box, person_id, confidence in zip(boxes, ids, confidences):

                        x1, y1, x2, y2 = box

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        box_width = x2 - x1
                        box_height = y2 - y1

                        current_time = frame_count / fps

                        speed = 0.0
                        acceleration = 0.0

                        if person_id in history:

                            previous = history[person_id]

                            elapsed = current_time - previous["time"]

                            if elapsed > 0:

                                distance = math.hypot(
                                    center_x - previous["x"], center_y - previous["y"]
                                )

                                speed = distance / elapsed

                                acceleration = (speed - previous["speed"]) / elapsed

                        history[person_id] = {
                            "x": center_x,
                            "y": center_y,
                            "time": current_time,
                            "speed": speed,
                        }

                        csv_writer.writerow(
                            [
                                frame_count,
                                round(current_time, 3),
                                person_id,
                                round(center_x, 2),
                                round(center_y, 2),
                                round(box_width, 2),
                                round(box_height, 2),
                                round(confidence, 4),
                                round(speed, 2),
                                round(acceleration, 2),
                            ]
                        )

                writer.write(annotated_frame)

                frame_count += 1

                if frame_count % 30 == 0:
                    print(f"Fotogramas procesados: {frame_count}")

        finally:
            cap.release()
            writer.release()

    print("\nProcesamiento finalizado.")
    print(f"Fotogramas: {frame_count}")
    print(f"Video: {output_path}")
    print(f"Trayectorias CSV: {csv_path}")


if __name__ == "__main__":
    main()
