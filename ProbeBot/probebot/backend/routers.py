from fastapi import APIRouter, File, UploadFile, Request
from probeMeasurement import measureTip
from backend.entities import MeasurementResponse
import logging
import numpy as np
import os
import cv2


bot_router = APIRouter(prefix="", tags=[""])
logger = logging.getLogger("uvicorn.error")


@bot_router.post(
    "/measure",
    response_model=MeasurementResponse,
    status_code=200,
    description="Measure a probe image.",
)
async def measure_image_wrapper(request: Request):
    form_data = await request.form()
    orientation = form_data.get("orientation")
    image = form_data.get("image")
    result = await measure_image(orientation, image)
    return result


async def measure_image(orientation: str, image: UploadFile = File(...)):

    if os.path.exists("backend/temp/temp.tif"):
        os.remove("backend/temp/temp.tif")

    contents = await image.read()
    with open(f"backend/temp/temp.tif", "wb") as f:
        f.write(contents)

    measurement = measureTip(orientation)
    return MeasurementResponse(image=measurement[0], radius=measurement[1])
