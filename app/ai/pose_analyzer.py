import math
from collections import deque


class PoseAnalyzer:

    KEYPOINTS = {
        "nose": 0,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    def __init__(self, confidence_threshold=0.25, tracking_gap_seconds=1.0):

        self.confidence_threshold = confidence_threshold
        self.tracking_gap_seconds = tracking_gap_seconds

        # Historial independiente para cada ID de ByteTrack
        self.person_history = {}

        self.window_size = 10
        self.minimum_samples = 5
        self.movement_threshold = 15.0

    @staticmethod
    def calculate_distance(point_a, point_b):

        return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])

    @staticmethod
    def calculate_angle(point_a, point_b, point_c):

        # El punto B es la articulación central
        vector_1 = (point_a[0] - point_b[0], point_a[1] - point_b[1])

        vector_2 = (point_c[0] - point_b[0], point_c[1] - point_b[1])

        magnitude_1 = math.hypot(*vector_1)
        magnitude_2 = math.hypot(*vector_2)

        if magnitude_1 == 0 or magnitude_2 == 0:
            return None

        dot_product = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]

        cosine = dot_product / (magnitude_1 * magnitude_2)

        cosine = max(-1.0, min(1.0, cosine))

        return math.degrees(math.acos(cosine))

    def get_point(self, points, confidences, index):

        x, y = points[index]

        confidence = confidences[index]

        if confidence < self.confidence_threshold:
            return None

        if x < 0 and y < 0:
            return None

        return (float(x), float(y))

    def calculate_joint_angle(self, points, confidences, index_a, index_b, index_c):

        point_a = self.get_point(points, confidences, index_a)
        point_b = self.get_point(points, confidences, index_b)
        point_c = self.get_point(points, confidences, index_c)

        if any(point is None for point in (point_a, point_b, point_c)):
            return None

        return self.calculate_angle(point_a, point_b, point_c)

    def calculate_torso_angle(self, points, confidences):

        left_shoulder = self.get_point(points, confidences, 5)

        right_shoulder = self.get_point(points, confidences, 6)

        left_hip = self.get_point(points, confidences, 11)

        right_hip = self.get_point(points, confidences, 12)

        if any(
            point is None
            for point in (left_shoulder, right_shoulder, left_hip, right_hip)
        ):
            return None

        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )

        hip_center = (
            (left_hip[0] + right_hip[0]) / 2,
            (left_hip[1] + right_hip[1]) / 2,
        )

        # Ángulo del torso respecto a la vertical de la imagen
        dx = shoulder_center[0] - hip_center[0]
        dy = hip_center[1] - shoulder_center[1]

        angle = math.degrees(math.atan2(dx, dy))

        return abs(angle)

    def calculate_body_center(self, points, confidences):

        indexes = [5, 6, 11, 12]

        valid_points = []

        for index in indexes:

            point = self.get_point(points, confidences, index)

            if point is not None:
                valid_points.append(point)

        # Se requieren al menos tres puntos del torso
        if len(valid_points) < 3:
            return None

        center_x = sum(p[0] for p in valid_points) / len(valid_points)
        center_y = sum(p[1] for p in valid_points) / len(valid_points)

        return (center_x, center_y)

    def analyze_movement(self, track_id, body_center, current_time):

        if body_center is None:
            return {
                "displacement_px": None,
                "velocity_px_sec": None,
                "movement_state": "SIN_REFERENCIA",
            }

        if track_id not in self.person_history:

            self.person_history[track_id] = deque(maxlen=self.window_size)

        history = self.person_history[track_id]

        # Comprobar interrupción del seguimiento
        if history:

            elapsed = current_time - history[-1]["time"]

            if elapsed <= 0 or elapsed > self.tracking_gap_seconds:
                history.clear()

        # Registrar posición corporal
        history.append({"time": current_time, "x": body_center[0], "y": body_center[1]})

        if len(history) < self.minimum_samples:

            return {
                "displacement_px": None,
                "velocity_px_sec": None,
                "movement_state": "SIN_REFERENCIA",
            }

        # Regresión lineal de las posiciones respecto al tiempo
        times = [item["time"] for item in history]
        positions_x = [item["x"] for item in history]
        positions_y = [item["y"] for item in history]

        mean_time = sum(times) / len(times)
        mean_x = sum(positions_x) / len(positions_x)
        mean_y = sum(positions_y) / len(positions_y)

        denominator = sum((t - mean_time) ** 2 for t in times)

        if denominator <= 0:

            return {
                "displacement_px": None,
                "velocity_px_sec": None,
                "movement_state": "SIN_REFERENCIA",
            }

        velocity_x = (
            sum(
                (times[i] - mean_time) * (positions_x[i] - mean_x)
                for i in range(len(times))
            )
            / denominator
        )

        velocity_y = (
            sum(
                (times[i] - mean_time) * (positions_y[i] - mean_y)
                for i in range(len(times))
            )
            / denominator
        )

        velocity = math.hypot(velocity_x, velocity_y)

        displacement = math.hypot(
            positions_x[-1] - positions_x[0], positions_y[-1] - positions_y[0]
        )

        if velocity >= self.movement_threshold:
            movement_state = "EN_MOVIMIENTO"
        else:
            movement_state = "MOVIMIENTO_BAJO"

        return {
            "displacement_px": round(displacement, 2),
            "velocity_px_sec": round(velocity, 2),
            "movement_state": movement_state,
        }

    def analyze(self, track_id, points, confidences, current_time):

        angles = {
            "left_elbow": self.calculate_joint_angle(points, confidences, 5, 7, 9),
            "right_elbow": self.calculate_joint_angle(points, confidences, 6, 8, 10),
            "left_knee": self.calculate_joint_angle(points, confidences, 11, 13, 15),
            "right_knee": self.calculate_joint_angle(points, confidences, 12, 14, 16),
            "torso": self.calculate_torso_angle(points, confidences),
        }

        angles = {
            name: round(value, 2) if value is not None else None
            for name, value in angles.items()
        }

        body_center = self.calculate_body_center(points, confidences)
        keypoints = {}

        for name, index in self.KEYPOINTS.items():
            point = self.get_point(points, confidences, index)
            keypoints[name] = point

        movement = self.analyze_movement(track_id, body_center, current_time)

        return {
            "track_id": track_id,
            "angles": angles,
            "keypoints": keypoints,
            "body_center": body_center,
            "movement": movement,
        }
