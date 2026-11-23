from app.video.video_reader import VideoReader

VIDEO_PATH = "recordings/input/test.mp4"


def main():

    video = VideoReader(VIDEO_PATH)

    try:

        information = video.get_information()

        print("\nINFORMACION DEL VIDEO")
        print("------------------------------")

        for key, value in information.items():
            print(f"{key}: {value}")

        print("\nReproduciendo video...")

        while True:

            frame = video.read_frame()

            if frame is None:
                break

            cv2_frame = frame.copy()

            import cv2

            cv2.imshow("Digital Surveillance System", cv2_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        video.release()

        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
