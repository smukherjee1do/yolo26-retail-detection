from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


def run_inference(image_path):

    model = YOLO("models/best.pt")

    results = model(image_path)

    result = results[0]

    annotated = result.plot()

    plt.imshow(annotated)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    run_inference("images/image1.jpeg")
