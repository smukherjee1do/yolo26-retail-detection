from ultralytics import YOLO

def train_model():

    # Load pretrained YOLO26 model
    model = YOLO("yolo26n.pt")

    # Train on SKU dataset
    results = model.train(
        data="data/SKU-110K.yaml",
        epochs=100,
        imgsz=640
    )

    return results


if __name__ == "__main__":
    train_model()
