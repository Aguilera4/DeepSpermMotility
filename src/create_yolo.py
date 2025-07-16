import torch
from ultralytics import YOLO

model = YOLO("./runs/detect/train7/weights/best.pt")
#results = model.train(data="visem-tracing.yaml", epochs=100, imgsz=640)

#model = torch.hub.load('ultralytics/yolov5', 'custom', path='../YOLO_model/best_yolov5x.pt')

try:
    #results = model("../data/VISEM_Tracking/train/11/images/11_frame_0.jpg")
    results = model.track("../data/VISEM_Tracking/train/11/11.mp4", show=True)
    print(results[0].show())
    bbox_data = results.pandas().xyxy[0]
    detections = bbox_data[['xmin', 'ymin', 'xmax', 'ymax', 'confidence']].values

    print(detections)
    
except Exception as e:
    print(e)