from ultralytics import YOLO

def predict(source, model_name="models/pretrained/yolov8n.pt", threshold=0.25, project="data", name="results"):
    model = YOLO(model_name)
    results = model.predict(source=source, conf=threshold, project=project, name=name, save=True, save_txt=True, exist_ok=True, imgsz=640)
    return results
