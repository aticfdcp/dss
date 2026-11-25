from pathlib import Path
import csv
import cv2

from pose_detector import PoseDetector


def main():

    project_root = Path(__file__).resolve().parents[2]

    video_path = project_root / "recordings" / "input" / "test.mp4"

    output_dir = project_root / "logs" / "pose_tracking"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_video = output_dir / "pose_tracking.mp4"
    output_csv = output_dir / "pose_tracking.csv"

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("No se pudo crear el video de salida.")

    detector = PoseDetector()

    # Encabezados del CSV
    headers = [
        "fotograma",
        "tiempo_segundos",
        "id_persona",
        "confianza_deteccion",
        "x1",
        "y1",
        "x2",
        "y2",
    ]

    for i in range(17):
        headers.extend([f"punto_{i}_x", f"punto_{i}_y", f"punto_{i}_conf"])

    frame_number = 0
    unique_ids = set()

    try:

        with open(output_csv, "w", newline="", encoding="utf-8-sig") as csv_file:

            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)

            while frame_number < 300:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_number += 1

                result = detector.track(frame)

                annotated_frame = result.plot()

                writer.write(annotated_frame)

                boxes = result.boxes
                keypoints = result.keypoints

                person_count = len(boxes) if boxes is not None else 0

                if person_count > 0 and keypoints is not None:

                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    boxes_conf = boxes.conf.cpu().numpy()

                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * person_count

                    points_xy = keypoints.xy.cpu().numpy()

                    if keypoints.conf is not None:
                        points_conf = keypoints.conf.cpu().numpy()
                    else:
                        points_conf = None

                    for index in range(person_count):

                        track_id = track_ids[index]

                        if track_id is not None:
                            unique_ids.add(track_id)

                        x1, y1, x2, y2 = boxes_xyxy[index]

                        row = [
                            frame_number,
                            round(frame_number / fps, 3),
                            track_id,
                            round(float(boxes_conf[index]), 4),
                            round(float(x1), 2),
                            round(float(y1), 2),
                            round(float(x2), 2),
                            round(float(y2), 2),
                        ]

                        for point_index in range(17):

                            x, y = points_xy[index][point_index]

                            if points_conf is not None:
                                confidence = points_conf[index][point_index]
                            else:
                                confidence = ""

                            row.extend(
                                [
                                    round(float(x), 2),
                                    round(float(y), 2),
                                    (
                                        round(float(confidence), 4)
                                        if confidence != ""
                                        else ""
                                    ),
                                ]
                            )

                        csv_writer.writerow(row)

                if frame_number % 30 == 0:

                    print(
                        f"Fotogramas: {frame_number} | "
                        f"Personas detectadas: {person_count} | "
                        f"IDs acumulados: {len(unique_ids)}"
                    )

    finally:

        cap.release()
        writer.release()

    print("\nPrueba finalizada.")
    print(f"Fotogramas procesados: {frame_number}")
    print(f"IDs de seguimiento diferentes: {len(unique_ids)}")
    print(f"Video generado: {output_video}")
    print(f"CSV generado: {output_csv}")


if __name__ == "__main__":
    main()
