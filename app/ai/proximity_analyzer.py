import math


class ProximityAnalyzer:

    def __init__(self, proximity_threshold=250):
        """
        proximity_threshold:
            Distancia en píxeles para considerar
            que dos personas están próximas.
        """

        self.proximity_threshold = proximity_threshold

    def calculate_distance(self, person_a, person_b):
        """
        Calcula la distancia euclidiana entre
        los centros de dos personas.
        """

        return math.hypot(
            person_a["center_x"] - person_b["center_x"],
            person_a["center_y"] - person_b["center_y"],
        )

    def analyze(self, people):
        """
        Analiza las distancias entre todas
        las parejas de personas detectadas.
        """

        results = []

        for i in range(len(people)):

            for j in range(i + 1, len(people)):

                person_a = people[i]
                person_b = people[j]

                distance = self.calculate_distance(person_a, person_b)

                is_close = distance <= self.proximity_threshold

                results.append(
                    {
                        "person_a": person_a["id"],
                        "person_b": person_b["id"],
                        "distance": round(distance, 2),
                        "is_close": is_close,
                    }
                )

        return results


if __name__ == "__main__":

    analyzer = ProximityAnalyzer(proximity_threshold=250)

    # Datos simulados de tres personas
    people = [
        {"id": 1, "center_x": 400, "center_y": 300},
        {"id": 2, "center_x": 550, "center_y": 320},
        {"id": 3, "center_x": 1000, "center_y": 500},
    ]

    results = analyzer.analyze(people)

    print("\nANÁLISIS DE PROXIMIDAD")
    print("----------------------")

    for result in results:

        status = "PRÓXIMAS" if result["is_close"] else "SEPARADAS"

        print(
            f"Persona {result['person_a']} - "
            f"Persona {result['person_b']} | "
            f"Distancia: {result['distance']} px | "
            f"Estado: {status}"
        )
