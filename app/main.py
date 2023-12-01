from predict import predict

if __name__ == "__main__":
    print(predict(source="D:\Projects\yolo_v8\sources\street_2.jpeg", model_name="models/pretrained/yolov8n.pt", threshold=0.5))