from ultralytics import YOLOv10

import cv2
import supervision as sv
from ultralytics import YOLOv10

model = YOLOv10(f'weights/yolov10n.pt')
image = cv2.imread(f'konten/gambar.jpg')
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

sv.plot_image(annotated_image)
