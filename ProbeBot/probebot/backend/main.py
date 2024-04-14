from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from probebot.backend.routers import bot_router
from probeMeasurement import MeasurementErrorException

app = FastAPI(
    title="Probe Bot",
    description="Automatic error detection and measurement of AFM Probes using AI and image processing tools.",
    version="0.1.0",
)


app.include_router(bot_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Exceptions --- #


@app.exception_handler(MeasurementErrorException)
def handle_measurement_error(
    _request: Request, exception: MeasurementErrorException
) -> JSONResponse:
    return JSONResponse(
        status_code=406,
        content={
            "detail": {
                "type": "measurement_error",
                "message": "The given probe image could not be processed.\n Please ensure that the input probe orientation is correct and the probe tip is not slanted.",
                "exception": exception,
            }
        },
    )
