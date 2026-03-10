import gradio as gr
from ultralytics import YOLO
import pandas as pd

model = YOLO("models/best.pt")


def detect_objects(image):

    results = model(image)

    annotated_img = results[0].plot()

    total_objects = len(results[0].boxes)

    class_counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]

        if class_name not in class_counts:
            class_counts[class_name] = 0

        class_counts[class_name] += 1

    df = pd.DataFrame(
        list(class_counts.items()),
        columns=["Class", "Count"]
    )

    return annotated_img, total_objects, df


app = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(label="Predicted Image"),
        gr.Number(label="Total Objects Detected"),
        gr.Dataframe(label="Detected Class Counts"),
    ],
    title="YOLO26 Retail Object Detection",
    description="Upload an image to detect products using a fine-tuned YOLO26 model."
)

app.launch()
