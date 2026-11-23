from interaction_analyzer import InteractionAnalyzer

analyzer = InteractionAnalyzer(
    sustained_proximity_seconds=2.0, distance_change_threshold=30.0
)

# Simulación de una pareja de personas
test_data = [
    {"time": 0.0, "distance": 300, "close": False},
    {"time": 1.0, "distance": 260, "close": False},
    {"time": 2.0, "distance": 220, "close": True},
    {"time": 3.0, "distance": 180, "close": True},
    {"time": 4.0, "distance": 170, "close": True},
    {"time": 5.0, "distance": 300, "close": False},
]

for sample in test_data:

    proximity_results = [
        {
            "person_a": 1,
            "person_b": 2,
            "distance": sample["distance"],
            "is_close": sample["close"],
        }
    ]

    result = analyzer.analyze(proximity_results, sample["time"])

    interaction = result[0]

    print(
        f"Tiempo: {sample['time']} s | "
        f"Distancia: {interaction['distance']} px | "
        f"Proximidad: {interaction['proximity_state']} | "
        f"Movimiento: {interaction['distance_state']} | "
        f"Duración: {interaction['proximity_duration']} s | "
        f"Interacción: {interaction['interaction_state']}"
    )
