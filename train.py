from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="C:/Users/PC-1/Desktop/project/dataset/data.yaml", epochs=20, 
    imgsz=256, 
    batch=16, 
    mosaic=1.0,
    mixup=0.2,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5
    )