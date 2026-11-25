from pathlib import Path
import csv
import cv2

from pose_detector import PoseDetector
from pose_analyzer import PoseAnalyzer


def main():

    project_root = Path(__file__).resolve().parents[2]

    video_path = project_root / "recordings" / "input" / "test.mp4"

    output_dir = project_root / "logs" / "pose_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "pose_analysis.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    detector = PoseDetector()
    analyzer = PoseAnalyzer(confidence_threshold=0.25, tracking_gap_seconds=1.0)

    # Articulaciones y nombres de sus coordenadas
    keypoint_names = list(PoseAnalyzer.KEYPOINTS.keys())

    keypoint_headers = []

    for name in keypoint_names:
        keypoint_headers.extend([f"{name}_x", f"{name}_y"])

    headers = [
        "fotograma",
        "tiempo_segundos",
        "id_persona",
        "angulo_codo_izquierdo",
        "angulo_codo_derecho",
        "angulo_rodilla_izquierda",
        "angulo_rodilla_derecha",
        "angulo_torso",
        "centro_x",
        "centro_y",
        "desplazamiento_px",
        "velocidad_px_seg",
        "estado_movimiento",
    ] + keypoint_headers

    frame_number = 0
    total_person_records = 0

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as csv_file:

        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(headers)

        try:

            while frame_number < 300:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                current_time = (frame_number - 1) / fps

                result = detector.track(frame)

                boxes = result.boxes
                keypoints = result.keypoints

                if boxes is None or boxes.id is None or keypoints is None:
                    continue

                track_ids = boxes.id.int().cpu().tolist()

                points_xy = keypoints.xy.cpu().numpy()

                if keypoints.conf is None:
                    continue

                points_conf = keypoints.conf.cpu().numpy()

                for index, track_id in enumerate(track_ids):

                    analysis = analyzer.analyze(
                        track_id=track_id,
                        points=points_xy[index],
                        confidences=points_conf[index],
                        current_time=current_time,
                    )

                    angles = analysis["angles"]
                    movement = analysis["movement"]
                    center = analysis["body_center"]
                    keypoints_data = analysis["keypoints"]

                    center_x = center[0] if center else ""
                    center_y = center[1] if center else ""

                    row = [
                        frame_number,
                        round(current_time, 3),
                        track_id,
                        angles["left_elbow"],
                        angles["right_elbow"],
                        angles["left_knee"],
                        angles["right_knee"],
                        angles["torso"],
                        round(center_x, 2) if center_x != "" else "",
                        round(center_y, 2) if center_y != "" else "",
                        movement["displacement_px"],
                        movement["velocity_px_sec"],
                        movement["movement_state"],
                    ]

                    # Incorporar coordenadas X e Y de cada articulación
                    for name in keypoint_names:

                        point = keypoints_data.get(name)

                        if point is not None:
                            row.extend([round(point[0], 2), round(point[1], 2)])
                        else:
                            row.extend(["", ""])

                    csv_writer.writerow(row)

                    total_person_records += 1

                if frame_number % 30 == 0:

                    print(
                        f"Fotogramas: {frame_number} | "
                        f"Registros corporales: {total_person_records}"
                    )

        finally:

            cap.release()

    print("\nAnálisis corporal finalizado.")
    print(f"Fotogramas procesados: {frame_number}")
    print(f"Registros de personas: {total_person_records}")
    print(f"Total de columnas CSV: {len(headers)}")
    print(f"CSV generado: {output_csv}")


if __name__ == "__main__":
    main()
