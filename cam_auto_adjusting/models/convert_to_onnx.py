from ultralytics import YOLO

model = YOLO("best1_with_corner.pt")

model.export(format="onnx")

