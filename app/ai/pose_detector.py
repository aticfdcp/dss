from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


class PoseDetector:

    def __init__(self):

        self.project_root = Path(__file__).resolve().parents[2]

        self.model_path = self.project_root / "models" / "yolo11n-pose.pt"

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.device = 0 if torch.cuda.is_available() else "cpu"

        # Comprobar si el modelo ya existe dentro del proyecto
        if self.model_path.exists():

            self.model = YOLO(str(self.model_path))

        else:

            # Comprobar si fue descargado en el directorio actual
            local_model = Path.cwd() / "yolo11n-pose.pt"

            if local_model.exists():

                shutil.copy2(local_model, self.model_path)

                self.model = YOLO(str(self.model_path))

            else:

                print("Descargando modelo YOLO11n Pose...")

                self.model = YOLO("yolo11n-pose.pt")

                downloaded_model = Path(self.model.ckpt_path)

                if downloaded_model.exists():

                    shutil.copy2(downloaded_model, self.model_path)

                    self.model = YOLO(str(self.model_path))

        print("Modelo YOLO11n Pose cargado")

        print(f"Dispositivo: " f"{'GPU' if self.device == 0 else 'CPU'}")

    def track(self, frame):

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.15,
            iou=0.50,
            imgsz=960,
            device=self.device,
            verbose=False,
        )

        return results[0]
