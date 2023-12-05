import uvicorn
import os
from actions import predict
from fastapi import FastAPI, UploadFile
from typing import List

app = FastAPI()

@app.post("/actions/predictImage")
async def predict_image(images: List[UploadFile] = UploadFile(...)):
    os.makedirs("data/uploads", exist_ok=True)

    for id, image in enumerate(images, start=1):
        file_path = os.path.join("data/uploads", image.filename)

        with open(file_path, "wb") as f:
            f.write(image.file.read())

        predict(source=file_path, model_name="models/pretrained/yolov8n.pt", threshold=0.5)
        print(f"Image {id}/{len(images)}: Processed")

    return "SUCCESS"

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)