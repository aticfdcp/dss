from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
import math


class UnusualInteractionDetector:
    """
    Detecta patrones candidatos de interacción inusual entre personas.
    No determina agresiones ni intenciones: los resultados requieren revisión humana.
    """

    def __init__(
        self,
        score_threshold: float = 0.55,
        minimum_candidate_seconds: float = 0.35,
        event_end_grace_seconds: float = 1.0,
        tracking_gap_seconds: float = 1.0,
        history_seconds: float = 1.0,
        wrist_speed_threshold: float = 1.8,
        body_speed_threshold: float = 1.2,
        cooldown_seconds: float = 3.0,
    ):
        self.score_threshold = score_threshold
        self.minimum_candidate_seconds = minimum_candidate_seconds
        self.event_end_grace_seconds = event_end_grace_seconds
        self.tracking_gap_seconds = tracking_gap_seconds
        self.history_seconds = history_seconds
        self.wrist_speed_threshold = wrist_speed_threshold
        self.body_speed_threshold = body_speed_threshold
        self.cooldown_seconds = cooldown_seconds
        self.person_history: Dict[int, Deque[dict]] = {}
        self.pair_history: Dict[Tuple[int, int], dict] = {}
        self.next_event_number = 1

    @staticmethod
    def _point(value: Any) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        try:
            x, y = float(value[0]), float(value[1])
            if math.isfinite(x) and math.isfinite(y) and x >= 0 and y >= 0:
                return x, y
        except (TypeError, ValueError, IndexError):
            pass
        return None

    @staticmethod
    def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _get_track_id(person: dict) -> Optional[int]:
        value = person.get("track_id", person.get("id"))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _get_speed(movement: Any) -> Optional[float]:
        if not isinstance(movement, dict):
            return None
        for key in (
            "speed", "velocity", "velocity_px_sec",
            "velocidad_px_seg", "speed_px_s"
        ):
            value = movement.get(key)
            if value is not None:
                try:
                    value = float(value)
                    return value if math.isfinite(value) else None
                except (TypeError, ValueError):
                    pass
        return None

    def _body_scale(self, keypoints: dict) -> Optional[float]:
        ls = self._point(keypoints.get("left_shoulder"))
        rs = self._point(keypoints.get("right_shoulder"))
        lh = self._point(keypoints.get("left_hip"))
        rh = self._point(keypoints.get("right_hip"))

        shoulder_width = self._distance(ls, rs) if ls and rs else None
        shoulder_mid = (
            ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            if ls and rs else None
        )
        hip_mid = (
            ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            if lh and rh else None
        )
        torso_length = (
            self._distance(shoulder_mid, hip_mid)
            if shoulder_mid and hip_mid else None
        )
        candidates = [
            value for value in (shoulder_width, torso_length)
            if value is not None and value >= 8.0
        ]
        return max(candidates) if candidates else None

    def _update_person_motion(
        self, pose_results: Iterable[dict], current_time: float
    ) -> Dict[int, dict]:
        motion_by_id: Dict[int, dict] = {}

        for person in pose_results or []:
            track_id = self._get_track_id(person)
            if track_id is None:
                continue

            keypoints = person.get("keypoints") or {}
            if not isinstance(keypoints, dict):
                keypoints = {}

            wrists = [
                self._point(keypoints.get("left_wrist")),
                self._point(keypoints.get("right_wrist")),
            ]
            wrists = [point for point in wrists if point is not None]
            body_center = self._point(person.get("body_center"))
            scale = self._body_scale(keypoints)

            history = self.person_history.setdefault(track_id, deque())
            history.append({
                "time": current_time,
                "wrists": wrists,
                "body_center": body_center,
                "scale": scale,
            })

            while history and current_time - history[0]["time"] > self.history_seconds:
                history.popleft()

            wrist_speed_norm = None
            body_speed_norm = None

            if len(history) >= 2:
                previous = history[-2]
                dt = current_time - previous["time"]
                if 0 < dt <= self.tracking_gap_seconds:
                    reference_scale = scale or previous["scale"] or 1.0

                    if wrists and previous["wrists"]:
                        distances = [
                            self._distance(now, old)
                            for now in wrists for old in previous["wrists"]
                        ]
                        wrist_speed_norm = min(distances) / dt / reference_scale

                    if body_center and previous["body_center"]:
                        body_speed_norm = (
                            self._distance(body_center, previous["body_center"])
                            / dt / reference_scale
                        )

            supplied_speed = self._get_speed(person.get("movement"))
            if body_speed_norm is None and supplied_speed is not None and scale:
                body_speed_norm = supplied_speed / scale

            motion_by_id[track_id] = {
                "fast_wrist_motion": (
                    wrist_speed_norm is not None
                    and wrist_speed_norm >= self.wrist_speed_threshold
                ),
                "fast_body_motion": (
                    body_speed_norm is not None
                    and body_speed_norm >= self.body_speed_threshold
                ),
                "wrist_speed_norm": wrist_speed_norm,
                "body_speed_norm": body_speed_norm,
                "scale_px": scale,
            }

        stale_ids = [
            track_id for track_id, history in self.person_history.items()
            if not history or current_time - history[-1]["time"] > self.tracking_gap_seconds
        ]
        for track_id in stale_ids:
            del self.person_history[track_id]

        return motion_by_id

    def _score_pair(
        self, interaction: dict, motion_by_id: Dict[int, dict]
    ) -> Tuple[float, List[str]]:
        person_a = int(interaction["person_a"])
        person_b = int(interaction["person_b"])
        motion_a = motion_by_id.get(person_a, {})
        motion_b = motion_by_id.get(person_b, {})

        score = 0.0
        reasons: List[str] = []

        proximity_state = str(interaction.get("proximity_state", "")).upper()
        interaction_state = str(interaction.get("interaction_state", "")).upper()
        distance_state = str(interaction.get("distance_state", "")).upper()

        close = (
            proximity_state == "PROXIMAS"
            or interaction_state in (
                "PROXIMIDAD_RECIENTE", "PROXIMIDAD_MANTENIDA"
            )
        )
        if close:
            score += 0.25
            reasons.append("PERSONAS_CERCANAS")

        if distance_state == "ACERCANDOSE":
            score += 0.15
            reasons.append("ACERCAMIENTO")

        fast_a = motion_a.get("fast_wrist_motion", False)
        fast_b = motion_b.get("fast_wrist_motion", False)
        body_a = motion_a.get("fast_body_motion", False)
        body_b = motion_b.get("fast_body_motion", False)

        if fast_a or fast_b:
            score += 0.20
            reasons.append("MOVIMIENTO_RAPIDO_DE_BRAZOS")
        if fast_a and fast_b:
            score += 0.10
            reasons.append("MOVIMIENTO_RAPIDO_SIMULTANEO")
        if (fast_a and body_b) or (fast_b and body_a):
            score += 0.15
            reasons.append("MOVIMIENTO_CORPORAL_COMBINADO")
        if body_a or body_b:
            score += 0.10
            reasons.append("CAMBIO_RAPIDO_DEL_CENTRO_CORPORAL")
        if distance_state == "ALEJANDOSE" and close:
            score += 0.05
            reasons.append("SEPARACION_DESPUES_DE_PROXIMIDAD")

        return min(score, 1.0), reasons

    def analyze(
        self,
        pose_results: Iterable[dict],
        interaction_results: Iterable[dict],
        current_time: float,
    ) -> List[dict]:
        current_time = float(current_time)
        motion_by_id = self._update_person_motion(pose_results, current_time)
        output: List[dict] = []
        active_pairs = set()

        for interaction in interaction_results or []:
            try:
                person_a = int(interaction["person_a"])
                person_b = int(interaction["person_b"])
                pair_key = tuple(sorted((person_a, person_b)))
            except (KeyError, TypeError, ValueError):
                continue

            active_pairs.add(pair_key)
            score, reasons = self._score_pair(interaction, motion_by_id)

            state = self.pair_history.setdefault(pair_key, {
                "candidate_since": None,
                "last_candidate_time": None,
                "event_start": None,
                "event_id": None,
                "last_seen": current_time,
                "last_score": 0.0,
                "max_score": 0.0,
                "reasons": set(),
                "cooldown_until": 0.0,
            })
            state["last_seen"] = current_time
            state["last_score"] = score

            if score >= self.score_threshold:
                if state["candidate_since"] is None:
                    state["candidate_since"] = current_time
                state["last_candidate_time"] = current_time
                state["max_score"] = max(state["max_score"], score)
                state["reasons"].update(reasons)

                duration = current_time - state["candidate_since"]
                if (
                    state["event_start"] is None
                    and duration >= self.minimum_candidate_seconds
                    and current_time >= state["cooldown_until"]
                ):
                    state["event_start"] = state["candidate_since"]
                    state["event_id"] = f"EVT-{self.next_event_number:05d}"
                    self.next_event_number += 1

                if state["event_start"] is not None:
                    output.append(self._build_event(
                        pair_key, state, current_time, "EN_CURSO"
                    ))
            else:
                state["candidate_since"] = None
                if state["event_start"] is not None:
                    last_candidate = state["last_candidate_time"]
                    if (
                        last_candidate is not None
                        and current_time - last_candidate >= self.event_end_grace_seconds
                    ):
                        output.append(self._build_event(
                            pair_key, state, last_candidate, "FINALIZADO"
                        ))
                        self._clear_event(state, current_time)

        for pair_key, state in list(self.pair_history.items()):
            if pair_key in active_pairs:
                continue
            if current_time - state["last_seen"] > self.tracking_gap_seconds:
                if state["event_start"] is not None:
                    end_time = state["last_candidate_time"] or state["last_seen"]
                    output.append(self._build_event(
                        pair_key, state, end_time, "FINALIZADO"
                    ))
                del self.pair_history[pair_key]

        return output

    @staticmethod
    def _build_event(
        pair_key: Tuple[int, int], state: dict, end_time: float, status: str
    ) -> dict:
        return {
            "event_id": state["event_id"],
            "person_a": pair_key[0],
            "person_b": pair_key[1],
            "start_time": round(float(state["event_start"]), 3),
            "end_time": round(float(end_time), 3),
            "duration_seconds": round(
                max(0.0, float(end_time) - float(state["event_start"])), 3
            ),
            "score": round(float(state["last_score"]), 3),
            "max_score": round(float(state["max_score"]), 3),
            "reasons": sorted(state["reasons"]),
            "status": status,
            "classification": "INTERACCION_INUSUAL_CANDIDATA",
            "requires_human_review": True,
        }

    def _clear_event(self, state: dict, current_time: float) -> None:
        state["event_start"] = None
        state["event_id"] = None
        state["cooldown_until"] = current_time + self.cooldown_seconds
        state["max_score"] = 0.0
        state["reasons"] = set()
        state["candidate_since"] = None
        state["last_candidate_time"] = None

    def finalize(self, current_time: float) -> List[dict]:
        """Cierra los eventos que sigan activos al terminar el video."""
        output = []
        for pair_key, state in list(self.pair_history.items()):
            if state["event_start"] is not None:
                end_time = state["last_candidate_time"]
                if end_time is None:
                    end_time = current_time
                output.append(self._build_event(
                    pair_key, state, end_time, "FINALIZADO"
                ))
                self._clear_event(state, current_time)
        return output

    def reset(self) -> None:
        self.person_history.clear()
        self.pair_history.clear()
        self.next_event_number = 1
