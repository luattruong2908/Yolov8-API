import uvicorn
import os
from actions import predict_image_api, predict_video_api, allowed_file
from fastapi import FastAPI, UploadFile, HTTPException #, BackgroundTasks
from typing import List

app = FastAPI()

ml_status = False
is_running = 0

def file_process(file_type, files):
    os.makedirs(f"data/{file_type}/uploads", exist_ok=True)

    global ml_status
    global is_running
    ml_status = True
    is_running += 1

    for id, file in enumerate(files, start=1):
        if not allowed_file(file.filename, type=file_type):
            ml_status = False
            is_running -= 1

            raise HTTPException(status_code=400, detail="Wrong file type.")
        
        # Save file to specific location
        file_path = os.path.join(f"data/{file_type}/uploads", file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        if file_type == "image":
            predict_image_api(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.25)
            print(f"Image {id}/{len(files)}: Processed")
        elif file_type == "video":
            predict_video_api(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.25)
            print(f"Video {id}/{len(files)}: Processed")
        print(is_running)
        
    is_running -= 1
    if is_running == 0:
        ml_status = False

    print("DONE PROCESSING!!!")
    return None

@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/setup")
def setup():
    global ml_status

    return ml_status

@app.post("/actions/predict/{file_type}")
def predict(file_type: str, files: List[UploadFile]):
    file_process(file_type, files)

    return "PROCESSING..."
        
if __name__ == "__main__":
     uvicorn.run("main:app", host="localhost", port=8000, reload=True)

