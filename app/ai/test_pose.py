from pathlib import Path
import cv2

from pose_detector import PoseDetector


def main():

    project_root = Path(__file__).resolve().parents[2]

    video_path = project_root / "recordings" / "input" / "test.mp4"

    output_dir = project_root / "logs" / "pose"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "pose_test.jpg"

    # Abrir video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video.")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("No se pudo leer el primer fotograma.")

    # Inicializar detector
    detector = PoseDetector()

    # Detectar puntos corporales
    result = detector.detect(frame)

    # Número de personas detectadas
    person_count = len(result.keypoints) if result.keypoints is not None else 0

    print(f"Personas detectadas: {person_count}")

    if result.keypoints is not None:

        keypoints = result.keypoints.xy.cpu().numpy()

        print(f"Forma de matriz de puntos: {keypoints.shape}")

        for index, person_points in enumerate(keypoints):

            print(f"\nPersona {index + 1}")

            for point_index, point in enumerate(person_points):

                x, y = point

                print(f"Punto {point_index}: " f"X={x:.1f}, Y={y:.1f}")

    # Dibujar esqueleto y guardar resultado
    annotated_frame = result.plot()

    success = cv2.imwrite(str(output_path), annotated_frame)

    if not success:
        raise RuntimeError("No se pudo guardar la imagen.")

    print("\nPrueba finalizada.")
    print(f"Imagen generada: {output_path}")


if __name__ == "__main__":
    main()
