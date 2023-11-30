from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

def predict(source, model, threshold):
    model = model
    results = model.predict(source=source, conf=threshold, project="results", name="predict")