from pathlib import Path
import cv2
import csv
import math
from collections import deque

from person_tracker import PersonTracker
from proximity_analyzer import ProximityAnalyzer
from interaction_analyzer import InteractionAnalyzer


def calculate_velocity(history):
    """Calcula la velocidad mediante regresión lineal."""

    if len(history) < 5:
        return 0.0

    times = [item["time"] for item in history]
    xs = [item["x"] for item in history]
    ys = [item["y"] for item in history]

    mean_time = sum(times) / len(times)
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)

    denominator = sum((t - mean_time) ** 2 for t in times)

    if denominator == 0:
        return 0.0

    velocity_x = (
        sum((times[i] - mean_time) * (xs[i] - mean_x) for i in range(len(times)))
        / denominator
    )

    velocity_y = (
        sum((times[i] - mean_time) * (ys[i] - mean_y) for i in range(len(times)))
        / denominator
    )

    return math.hypot(velocity_x, velocity_y)


def calculate_smoothed_position(history):
    """Calcula la posición promedio de la ventana temporal."""

    x = sum(item["x"] for item in history) / len(history)
    y = sum(item["y"] for item in history) / len(history)

    return x, y


def main():

    project_root = Path(__file__).resolve().parents[2]

    video_path = project_root / "recordings" / "input" / "test.mp4"

    output_dir = project_root / "logs" / "proximity"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "proximity_analysis.csv"

    # Inicializar módulos
    tracker = PersonTracker()

    proximity_analyzer = ProximityAnalyzer(proximity_threshold=250)

    interaction_analyzer = InteractionAnalyzer(
        sustained_proximity_seconds=2.0, distance_change_threshold=30.0
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("FPS inválido.")

    # Configuración del análisis
    max_frames = 300
    window_size = 10

    movement_threshold = 5.0
    stationary_threshold = 3.0

    frame_count = 0

    person_history = {}
    person_states = {}
    pair_history = {}

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:

        writer = csv.writer(csv_file, delimiter=";")

        writer.writerow(
            [
                "fotograma",
                "tiempo_segundos",
                "tipo_registro",
                "persona_a",
                "persona_b",
                "posicion_x",
                "posicion_y",
                "velocidad_px_seg",
                "estado_movimiento",
                "distancia_px",
                "estado_proximidad",
                "variacion_distancia",
                "duracion_proximidad_seg",
                "estado_interaccion",
            ]
        )

        try:

            while frame_count < max_frames:

                ret, frame = cap.read()

                if not ret:
                    break

                result = tracker.track(frame)

                current_time = frame_count / fps
                people = []

                # -------------------------------------
                # 1. DETECCIÓN Y ANÁLISIS DE PERSONAS
                # -------------------------------------

                if result.boxes is not None and result.boxes.id is not None:

                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.int().cpu().tolist()

                    for box, person_id in zip(boxes, ids):

                        x1, y1, x2, y2 = box

                        foot_x = (x1 + x2) / 2
                        foot_y = y2

                        if person_id not in person_history:

                            person_history[person_id] = deque(maxlen=window_size)

                            person_states[person_id] = "SIN_REFERENCIA"

                        history = person_history[person_id]

                        # Detectar interrupciones del seguimiento
                        if history:

                            elapsed = current_time - history[-1]["time"]

                            if elapsed > 1.0:

                                history.clear()

                                person_states[person_id] = "SIN_REFERENCIA"

                        history.append(
                            {
                                "time": current_time,
                                "x": float(foot_x),
                                "y": float(foot_y),
                            }
                        )

                        # Suavizado de posición
                        smooth_x, smooth_y = calculate_smoothed_position(history)

                        # Cálculo de velocidad
                        speed = calculate_velocity(history)

                        previous_state = person_states[person_id]

                        # Clasificación de movimiento
                        if len(history) < 5:

                            movement_state = "SIN_REFERENCIA"

                        elif previous_state == "EN_MOVIMIENTO":

                            if speed < stationary_threshold:
                                movement_state = "ESTATICO"
                            else:
                                movement_state = "EN_MOVIMIENTO"

                        else:

                            if speed >= movement_threshold:
                                movement_state = "EN_MOVIMIENTO"
                            else:
                                movement_state = "ESTATICO"

                        person_states[person_id] = movement_state

                        people.append(
                            {
                                "id": person_id,
                                "center_x": smooth_x,
                                "center_y": smooth_y,
                            }
                        )

                        # Registro PERSONA
                        writer.writerow(
                            [
                                frame_count,
                                round(current_time, 3),
                                "PERSONA",
                                person_id,
                                "",
                                round(smooth_x, 2),
                                round(smooth_y, 2),
                                round(speed, 2),
                                movement_state,
                                "",
                                "",
                                "",
                                "",
                                "",
                            ]
                        )

                # -------------------------------------
                # 2. ANÁLISIS DE PROXIMIDAD
                # -------------------------------------

                proximity_results = proximity_analyzer.analyze(people)

                for item in proximity_results:

                    person_a = item["person_a"]
                    person_b = item["person_b"]
                    distance = item["distance"]

                    pair_key = tuple(sorted((person_a, person_b)))

                    distance_state = "SIN_REFERENCIA"

                    if pair_key in pair_history:

                        previous_distance = pair_history[pair_key]
                        difference = distance - previous_distance

                        if difference < -10:
                            distance_state = "ACERCANDOSE"

                        elif difference > 10:
                            distance_state = "ALEJANDOSE"

                        else:
                            distance_state = "ESTABLE"

                    pair_history[pair_key] = distance

                    proximity_state = "PROXIMAS" if item["is_close"] else "SEPARADAS"

                    # Registro PAREJA
                    writer.writerow(
                        [
                            frame_count,
                            round(current_time, 3),
                            "PAREJA",
                            person_a,
                            person_b,
                            "",
                            "",
                            "",
                            "",
                            round(distance, 2),
                            proximity_state,
                            distance_state,
                            "",
                            "",
                        ]
                    )

                # -------------------------------------
                # 3. ANÁLISIS DE INTERACCIONES
                # -------------------------------------

                interaction_results = interaction_analyzer.analyze(
                    proximity_results, current_time
                )

                for interaction in interaction_results:

                    writer.writerow(
                        [
                            frame_count,
                            round(current_time, 3),
                            "INTERACCION",
                            interaction["person_a"],
                            interaction["person_b"],
                            "",
                            "",
                            "",
                            "",
                            interaction["distance"],
                            interaction["proximity_state"],
                            interaction["distance_state"],
                            interaction["proximity_duration"],
                            interaction["interaction_state"],
                        ]
                    )

                # -------------------------------------
                # 4. PROGRESO
                # -------------------------------------

                frame_count += 1

                if frame_count % 30 == 0:

                    print(
                        f"Fotogramas: {frame_count} | "
                        f"Personas: {len(people)} | "
                        f"Parejas: {len(proximity_results)} | "
                        f"Interacciones: {len(interaction_results)}"
                    )

        finally:
            cap.release()

    print("\nAnálisis finalizado.")
    print(f"Fotogramas procesados: {frame_count}")
    print(f"CSV generado: {csv_path}")


if __name__ == "__main__":
    main()
