from typing import Annotated

from fastapi import APIRouter, File
from pydantic import BaseModel

from cocoa_classifier import predictor
from shared_state import state

prediction_router = APIRouter()


class PredictResponse(BaseModel):
    overlay: bytes
    results: list[predictor.PredictionResultRow]


@prediction_router.post(
    "/predict",
    response_model=PredictResponse,
)
async def predict(
    file: Annotated[bytes, File()],
    single_bean: bool = False,
):
    overlay, results = predictor.predict(
        file=file,
        model=state.model,
        classes=state.classes,
        single_bean=single_bean,
    )
    return PredictResponse(
        overlay=overlay.tobytes(),
        results=results,
    )
