from ultralytics import YOLO

model = YOLO(
    "C:/Users/PC-1/Desktop/project/curr_folder/ultralytics/cfg/models/v8/yolov8n_cbam.yaml"
)

model.train(
    data="C:/Users/PC-1/Desktop/project/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,

    lr0=0.003,
    lrf=0.01,

    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.1,
    scale=0.5,
    translate=0.1,

    box=7.5,
    cls=1.5,
    dfl=1.5
)
