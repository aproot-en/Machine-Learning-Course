


from ultralytics import YOLO
from pathlib import Path

def main():

    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="dataset/data.yaml",
        epochs=10,
        imgsz=480,
        batch=8,
        device=0,
        workers=0,
        amp=True,
        cache=False,
        project="runs",
        name="bdd100k",
        exist_ok=True,
        save=True,
        plots=True
    )

    # Output directory
    save_dir = Path(model.trainer.save_dir)

    print("Training completed!")
    print(f"Best Model: {save_dir / 'weights/best.pt'}")
    print(f"Last Model: {save_dir / 'weights/last.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
