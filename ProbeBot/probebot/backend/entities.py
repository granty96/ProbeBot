from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    BeforeValidator,
    PlainSerializer,
)
from typing_extensions import Annotated
import numpy as np


def nd_array_custom_before_validator(x):
    return x


def nd_array_custom_serializer(x):
    return str(x)


NdArray = Annotated[
    np.ndarray,
    BeforeValidator(nd_array_custom_before_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str),
]


class MeasurementResponse(BaseModel):
    radius: str
    image: NdArray

    model_config = ConfigDict(arbitrary_types_allowed=True)
