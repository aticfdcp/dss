from pathlib import Path
import csv

import cv2

from pose_detector import PoseDetector
from pose_analyzer import PoseAnalyzer
from proximity_analyzer import ProximityAnalyzer
from interaction_analyzer import InteractionAnalyzer
from unusual_interaction_detector import UnusualInteractionDetector


def main():
    project_root = Path(__file__).resolve().parents[2]
    video_path = project_root / "recordings" / "input" / "test.mp4"

    output_dir = project_root / "logs" / "pose_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pose_csv = output_dir / "pose_analysis.csv"
    output_events_csv = output_dir / "unusual_interaction_events.csv"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    pose_detector = PoseDetector()
    pose_analyzer = PoseAnalyzer(
        confidence_threshold=0.25,
        tracking_gap_seconds=1.0
    )
    proximity_analyzer = ProximityAnalyzer(proximity_threshold=250)
    interaction_analyzer = InteractionAnalyzer(
        sustained_proximity_seconds=2.0,
        distance_change_threshold=30.0,
        tracking_gap_seconds=1.0,
    )
    unusual_detector = UnusualInteractionDetector(
        score_threshold=0.55,
        minimum_candidate_seconds=0.35,
        event_end_grace_seconds=1.0,
    )

    keypoint_names = list(PoseAnalyzer.KEYPOINTS.keys())
    keypoint_headers = []
    for name in keypoint_names:
        keypoint_headers.extend([f"{name}_x", f"{name}_y"])

    pose_headers = [
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

    event_headers = [
        "id_evento",
        "persona_a",
        "persona_b",
        "tiempo_inicio_seg",
        "tiempo_fin_seg",
        "duracion_seg",
        "puntuacion_actual",
        "puntuacion_maxima",
        "indicadores",
        "estado",
        "clasificacion",
        "revision_humana",
    ]

    frame_number = 0
    total_person_records = 0
    latest_events = {}
    last_video_time = 0.0

    try:
        with (
            open(output_pose_csv, "w", newline="", encoding="utf-8-sig") as pose_file,
            open(output_events_csv, "w", newline="", encoding="utf-8-sig") as events_file,
        ):
            pose_writer = csv.writer(pose_file)
            event_writer = csv.writer(events_file)
            pose_writer.writerow(pose_headers)
            event_writer.writerow(event_headers)

            while frame_number < 300:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1
                current_time = (frame_number - 1) / fps
                last_video_time = current_time

                result = pose_detector.track(frame)
                boxes = result.boxes
                pose_keypoints = result.keypoints

                pose_results = []
                people_for_proximity = []

                if (
                    boxes is not None
                    and boxes.id is not None
                    and pose_keypoints is not None
                    and pose_keypoints.conf is not None
                ):
                    track_ids = boxes.id.int().cpu().tolist()
                    boxes_xyxy = boxes.xyxy.cpu().numpy()
                    points_xy = pose_keypoints.xy.cpu().numpy()
                    points_conf = pose_keypoints.conf.cpu().numpy()

                    for index, track_id in enumerate(track_ids):
                        analysis = pose_analyzer.analyze(
                            track_id=track_id,
                            points=points_xy[index],
                            confidences=points_conf[index],
                            current_time=current_time,
                        )

                        angles = analysis["angles"]
                        movement = analysis["movement"]
                        body_center = analysis["body_center"]
                        keypoints_data = analysis["keypoints"]

                        center_x = body_center[0] if body_center else ""
                        center_y = body_center[1] if body_center else ""

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

                        for name in keypoint_names:
                            point = keypoints_data.get(name)
                            if point is not None:
                                row.extend([
                                    round(point[0], 2),
                                    round(point[1], 2)
                                ])
                            else:
                                row.extend(["", ""])

                        pose_writer.writerow(row)
                        total_person_records += 1

                        # Formato esperado por UnusualInteractionDetector.
                        pose_results.append({
                            "track_id": track_id,
                            "keypoints": keypoints_data,
                            "body_center": body_center,
                            "movement": {
                                "velocity_px_sec": movement["velocity_px_sec"]
                            },
                        })

                        # Punto inferior central de la caja: aproximación a la
                        # posición de la persona en el plano de la imagen.
                        x1, y1, x2, y2 = boxes_xyxy[index]
                        people_for_proximity.append({
                            "id": track_id,
                            "center_x": float((x1 + x2) / 2.0),
                            "center_y": float(y2),
                        })

                # Relaciones espaciales entre todas las personas detectadas.
                proximity_results = proximity_analyzer.analyze(
                    people_for_proximity
                )

                # Evolución temporal de cada pareja.
                interaction_results = interaction_analyzer.analyze(
                    proximity_results,
                    current_time
                )

                # Detección de patrones candidatos combinando poses e interacción.
                detected_events = unusual_detector.analyze(
                    pose_results=pose_results,
                    interaction_results=interaction_results,
                    current_time=current_time,
                )

                # Guardamos la última versión de cada evento para no duplicar
                # una fila por cada fotograma mientras el evento está activo.
                for event in detected_events:
                    if event.get("event_id"):
                        latest_events[event["event_id"]] = event

                if frame_number % 30 == 0:
                    print(
                        f"Fotogramas: {frame_number} | "
                        f"Registros corporales: {total_person_records} | "
                        f"Eventos candidatos: {len(latest_events)}"
                    )

            # Cerrar cualquier evento que permanezca activo al final del video.
            final_events = unusual_detector.finalize(last_video_time)
            for event in final_events:
                if event.get("event_id"):
                    latest_events[event["event_id"]] = event

            # Escribir una sola fila por evento.
            for event_id in sorted(latest_events):
                event = latest_events[event_id]
                event_writer.writerow([
                    event.get("event_id"),
                    event.get("person_a"),
                    event.get("person_b"),
                    event.get("start_time"),
                    event.get("end_time"),
                    event.get("duration_seconds"),
                    event.get("score"),
                    event.get("max_score"),
                    ";".join(event.get("reasons", [])),
                    event.get("status"),
                    event.get("classification"),
                    event.get("requires_human_review"),
                ])

    finally:
        cap.release()

    print("\nAnálisis finalizado.")
    print(f"Fotogramas procesados: {frame_number}")
    print(f"Registros de personas: {total_person_records}")
    print(f"Columnas del CSV corporal: {len(pose_headers)}")
    print(f"Eventos candidatos: {len(latest_events)}")
    print(f"Análisis corporal: {output_pose_csv}")
    print(f"Eventos candidatos: {output_events_csv}")


if __name__ == "__main__":
    main()
