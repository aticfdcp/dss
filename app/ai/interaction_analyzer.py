class InteractionAnalyzer:

    def __init__(
        self,
        sustained_proximity_seconds=2.0,
        distance_change_threshold=30.0,
        tracking_gap_seconds=1.0,
    ):
        self.sustained_proximity_seconds = sustained_proximity_seconds
        self.distance_change_threshold = distance_change_threshold
        self.tracking_gap_seconds = tracking_gap_seconds

        self.pair_history = {}

    def analyze(self, proximity_results, current_time):

        interactions = []
        active_pairs = set()

        for item in proximity_results:

            person_a = item["person_a"]
            person_b = item["person_b"]

            distance = item["distance"]
            is_close = item["is_close"]

            pair_key = tuple(sorted((person_a, person_b)))
            active_pairs.add(pair_key)

            if pair_key not in self.pair_history:

                self.pair_history[pair_key] = {
                    "previous_distance": distance,
                    "close_since": current_time if is_close else None,
                    "last_seen": current_time,
                }

                distance_state = "SIN_REFERENCIA"

            else:

                history = self.pair_history[pair_key]

                elapsed = current_time - history["last_seen"]

                # Reiniciar referencias cuando se pierde el seguimiento
                if elapsed > self.tracking_gap_seconds:

                    history["previous_distance"] = distance

                    history["close_since"] = current_time if is_close else None

                    distance_state = "SIN_REFERENCIA"

                else:

                    distance_change = distance - history["previous_distance"]

                    change_rate = distance_change / elapsed if elapsed > 0 else 0

                    if change_rate <= -self.distance_change_threshold:

                        distance_state = "ACERCANDOSE"

                    elif change_rate >= self.distance_change_threshold:

                        distance_state = "ALEJANDOSE"

                    else:

                        distance_state = "ESTABLE"

                    if is_close:

                        if history["close_since"] is None:
                            history["close_since"] = current_time

                    else:

                        history["close_since"] = None

                    history["previous_distance"] = distance

                history["last_seen"] = current_time

            history = self.pair_history[pair_key]

            # Clasificación de proximidad
            if is_close:

                proximity_state = "PROXIMAS"

                proximity_duration = current_time - history["close_since"]

                if proximity_duration >= self.sustained_proximity_seconds:

                    interaction_state = "PROXIMIDAD_MANTENIDA"

                else:

                    interaction_state = "PROXIMIDAD_RECIENTE"

            else:

                proximity_state = "SEPARADAS"
                proximity_duration = 0.0

                interaction_state = "SIN_INTERACCION_CERCANA"

            interactions.append(
                {
                    "person_a": person_a,
                    "person_b": person_b,
                    "distance": round(distance, 2),
                    "proximity_state": proximity_state,
                    "distance_state": distance_state,
                    "proximity_duration": round(proximity_duration, 2),
                    "interaction_state": interaction_state,
                }
            )

        # Eliminar parejas que llevan demasiado tiempo sin detectarse
        stale_pairs = [
            pair_key
            for pair_key, history in self.pair_history.items()
            if (current_time - history["last_seen"] > self.tracking_gap_seconds)
            and pair_key not in active_pairs
        ]

        for pair_key in stale_pairs:
            del self.pair_history[pair_key]

        return interactions
