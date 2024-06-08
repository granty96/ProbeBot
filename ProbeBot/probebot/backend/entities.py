from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    BeforeValidator,
    PlainSerializer,
)
import json
from typing_extensions import Annotated
import numpy as np


def nd_array_custom_before_validator(x):
    return x


def nd_array_custom_serializer(x):
    return json.dumps(x.tolist())


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]


class MeasurementResponse(BaseModel):
    radius: str
    image: NdArray
    confidence: str
    prediction: str
    error: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
