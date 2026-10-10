
from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # Load trained model
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'bdd100k' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam (camera index 0)')

    try:
        while True:
            # Read frame
            ok, frame = cap.read()

            if not ok:
                break

            # Object detection
            results = model.predict(
                source=frame,
                conf=0.25,
                imgsz=640,
                device=0,
                verbose=False
            )

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Display result
            cv2.imshow(
                'BDD100K YOLO Real-time Detection',
                annotated_frame
            )

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
