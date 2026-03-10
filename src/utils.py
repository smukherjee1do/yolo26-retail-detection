from collections import Counter


def count_objects(result, model):

    detections = []

    for box in result.boxes:
        class_id = int(box.cls)
        label = model.names[class_id]
        detections.append(label)

    counts = Counter(detections)

    return counts
