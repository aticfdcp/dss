
from interaction_analyzer import InteractionAnalyzer

analyzer = InteractionAnalyzer(
    sustained_proximity_seconds=2.0,
    distance_change_threshold=30.0,
    tracking_gap_seconds=1.0
)

# Simulación de una pareja que se aproxima,
# permanece cerca y posteriormente se separa.
test_sequence = [
    (0.0, 400, False),
    (0.5, 300, True),
    (1.0, 280, True),
    (1.5, 270, True),
    (2.0, 260, True),
    (2.5, 250, True),
    (2.8, 300, False),
]

print("\nVALIDACIÓN DE INTERACTION ANALYZER\n")

for current_time, distance, is_close in test_sequence:

    proximity_results = [{
        "person_a": 1,
        "person_b": 2,
        "distance": distance,
        "is_close": is_close
    }]

    results = analyzer.analyze(
        proximity_results,
        current_time
    )

    for result in results:
        print(
            f"Tiempo: {current_time:4.1f}s | "
            f"Distancia: {result['distance']:6.2f}px | "
            f"Estado: {result['distance_state']:12} | "
            f"Proximidad: {result['interaction_state']:25} | "
            f"Duración: {result['proximity_duration']:.2f}s"
        )

# Simular pérdida de seguimiento durante más de un segundo
analyzer.analyze([], 4.0)

# La pareja vuelve a aparecer: debe iniciar una referencia nueva
results = analyzer.analyze(
    [{
        "person_a": 1,
        "person_b": 2,
        "distance": 200,
        "is_close": True
    }],
    4.1
)

print("\nVALIDACIÓN DESPUÉS DE PERDER EL SEGUIMIENTO")

for result in results:
    print(result)
