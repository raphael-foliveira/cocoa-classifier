import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from joblib import load

from routes import prediction_router
from shared_state import state

app = FastAPI()


@asynccontextmanager
async def lifespan(_):
    model_dir = Path("models/svm_v1")
    model = load("models/svm_v1/model-pkl")
    with open(model_dir / "classes.json") as f:
        classes = json.load(f)

    state.model = model
    state.classes = classes
    yield


app.include_router(prediction_router)
