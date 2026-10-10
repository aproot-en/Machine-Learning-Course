
from pathlib import Path
import random
import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # Model path
    weights = (
        ROOT / 'runs' / 'detect' / 'runs'
        / 'bdd100k' / 'weights' / 'best.pt'
    )

    # Test dataset
    image_dir = ROOT / 'dataset' / 'test' / 'images'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    if not image_dir.exists():
        raise FileNotFoundError(f'Dataset not found: {image_dir}')

    # Load trained model
    model = YOLO(str(weights))

    # Get test images
    extensions = {'.jpg', '.jpeg', '.png'}
    images = [
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    ]

    if not images:
        raise RuntimeError('No test images found')

    # Randomly select 5 images
    selected_images = random.sample(
        images, min(5, len(images))
    )

    # Object detection
    for image in selected_images:
        results = model.predict(
            source=str(image),
            conf=0.25,
            imgsz=480,
            device=0,
            save=True,
            project=str(ROOT / 'runs'),
            name='image_test',
            exist_ok=True
        )

        for result in results:
            print(f'\nImage: {image.name}')

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                print(
                    f'{model.names[class_id]}: '
                    f'{confidence:.2f}'
                )

            # Display detection result
            annotated = result.plot()
            cv2.imshow('YOLO Image Detection', annotated)

            # Press any key for next image
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print('\nImage testing completed!')


if __name__ == '__main__':
    main()
