from ultralytics import YOLO

model = YOLO("yolo11s.pt")
results = model.train(data="visem-tracing.yaml", epochs=100, imgsz=640)