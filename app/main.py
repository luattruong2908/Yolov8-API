import uvicorn
import os
from actions import predict_image_api, predict_video_api, allowed_file
from fastapi import FastAPI, UploadFile, HTTPException
from typing import List

app = FastAPI()

@app.post("/actions/predictImage")
async def predict_image(images: List[UploadFile] = UploadFile(...)):
    os.makedirs("data/image/uploads", exist_ok=True)

    for id, image in enumerate(images, start=1):
        if not allowed_file(image.filename, type='image'):
            raise HTTPException(status_code=400, detail="Wrong file type.")

        file_path = os.path.join("data/image/uploads", image.filename)

        with open(file_path, "wb") as f:
            f.write(image.file.read())

        predict_image_api(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.25)
        print(f"Image {id}/{len(images)}: Processed")

    return "SUCCESS"

@app.post("/actions/predictVideo")
async def predict_video(video: UploadFile = UploadFile(...)):
    os.makedirs("data/video/uploads", exist_ok=True)

    if not allowed_file(video.filename, type='video'):
            raise HTTPException(status_code=400, detail="Wrong file type.")
    file_path = os.path.join("data/video/uploads", video.filename)

    with open(file_path, "wb") as f:
        f.write(video.file.read())

    predict_video_api(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.25) 

    return "SUCCESS"

@app.post("/actions/predictImage")
async def predict_image(images: List[UploadFile] = UploadFile(...)):
    os.makedirs("data/image/uploads", exist_ok=True)

    for id, image in enumerate(images, start=1):
        if not allowed_file(image.filename, type='image'):
            raise HTTPException(status_code=400, detail="Wrong file type.")

        file_path = os.path.join("data/image/uploads", image.filename)

        with open(file_path, "wb") as f:
            f.write(image.file.read())

        predict_image_api(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.25)
        print(f"Image {id}/{len(images)}: Processed")

    return "SUCCESS"

if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=5000, reload=True)