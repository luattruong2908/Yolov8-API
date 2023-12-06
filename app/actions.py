from ultralytics import YOLO

def predict_image_api(source, model_name="models/pretrained/yolov8n.pt", threshold=0.25, project="data/image", name="results"):
    model = YOLO(model_name)
    results = model.predict(source=source, conf=threshold, project=project, name=name, save=True, save_txt=True, exist_ok=True, imgsz=640)
    return results

def predict_video_api(source, model_name="models/pretrained/yolov8n.pt", threshold=0.25, project="data/video", name="results"):
    model = YOLO(model_name)
    results = model.predict(source=source, conf=threshold, project=project, name=name, save=True, save_txt=True, exist_ok=True, imgsz=480, stream=False)  
    return results
