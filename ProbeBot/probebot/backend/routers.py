from fastapi import APIRouter, File, UploadFile
from probeMeasurement import measureTip
from backend.entities import MeasurementResponse
import os
import cv2


bot_router = APIRouter(prefix="", tags=[""])


@bot_router.post(
    "/measure",
    response_model=MeasurementResponse,
    status_code=200,
    description="Measure a probe image.",
)
async def measure_image(orientation: str, image: UploadFile = File(...)):

    if os.path.exists("backend/temp/temp.tif"):
        os.remove("backend/temp/temp.tif")

    contents = await image.read()
    with open(f"backend/temp/temp.tif", "wb") as f:
        f.write(contents)

    measurement = measureTip(orientation)
    return MeasurementResponse(image=measurement[0], radius=measurement[1])
