from fastapi import APIRouter, File, UploadFile, Request
from probebot.probeMeasurement import measureTip
from probebot.backend.entities import MeasurementResponse
from probebot.loadModelTorch import predict
from probebot.loadModelTorch import percentage
import logging
import traceback
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
    try:
        if os.path.exists("probebot/backend/temp/temp.tif"):
            os.remove("probebot/backend/temp/temp.tif")

        contents = await image.read()
        with open(f"probebot/backend/temp/temp.tif", "wb") as f:
            f.write(contents)

        prediction = predict()

        print(prediction)
        print(str(percentage(prediction[1], 3)))

        if prediction[0] != "Pass" or prediction[1] < 0.95:
            return MeasurementResponse(
                image=cv2.resize(
                    cv2.imread("probebot/backend/temp/temp.tif"), (800, 500)
                ),
                radius="",
                error="",
                prediction=prediction[0],
                confidence=str(percentage(prediction[1], 3)),
            )

        measurement = measureTip(orientation)

        return MeasurementResponse(
            image=measurement[0],
            radius=measurement[1],
            error="",
            prediction=prediction[0],
            confidence=str(prediction[1]),
        )

    except Exception as e:
        print(traceback.format_exc())
        return MeasurementResponse(
            image=np.zeros(1),
            radius="-1",
            error="The given probe image could not be processed.\n Please ensure that the input probe orientation is correct and the probe tip is not slanted",
            prediction="-1",
            confidence="-1",
        )
