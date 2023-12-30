import uvicorn
import os
from actions import predict_image_api, predict_video_api, allowed_file
from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from typing import List

app = FastAPI()

ml_status = False

async def file_process(file_type, files):
    os.makedirs(f"data/{file_type}/uploads", exist_ok=True)

    for id, file in enumerate(files, start=1):

        if not allowed_file(file.filename, type=file_type):
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
        
        global ml_status
        ml_status = False
    
    return "SUCCESS"

@app.get("/")
async def root():
    return {"Hello": "World"}

@app.get("/setup")
async def setup():
    global ml_status

    return ml_status

@app.post("/actions/predict/{file_type}")
async def predict(file_type: str, files: List[UploadFile]):
    global ml_status
    ml_status = True
    
    await file_process(file_type, files)

    return "SUCCESS"
        
if __name__ == "__main__":
     uvicorn.run("main:app", host="localhost", port=8000, workers=6)
