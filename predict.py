from ultralytics import YOLO

model = YOLO("pretrained/yolov8n.pt")

#def predict(source):
result = model.predict(source="1000_F_498170577_QlUqnkLW7vb4Ho5XmjgVTL6J1bMfBw8a.jpg", save=True, project="results", name="test")
    
#predict("shops-restaurants-around-yongkang-street-taipei-taiwan-dec-has-been-voted-one-coolest-streets-world-266468410.jpg")