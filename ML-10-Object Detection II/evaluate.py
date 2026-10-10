
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def main():
    # Load trained model
    weights = ROOT / 'runs' / 'detect' / 'runs' / 'bdd100k' / 'weights' / 'best.pt'

    if not weights.exists():
        raise FileNotFoundError(f'Model not found: {weights}')

    model = YOLO(str(weights))

    # Evaluate on test dataset
    metrics = model.val(
        data=str(ROOT / 'dataset' / 'data.yaml'),
        split='test',
        imgsz=480,
        batch=8,
        device=0,
        workers=0,
        plots=True,
        project=str(ROOT / 'runs'),
        name='evaluation',
        exist_ok=True,
    )

    # Display evaluation results
    print('\n===== Evaluation Results =====')
    print(f'Precision: {metrics.box.mp:.4f}')
    print(f'Recall: {metrics.box.mr:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP50-95: {metrics.box.map:.4f}')


if __name__ == '__main__':
    main()
