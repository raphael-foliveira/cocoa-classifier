from fastapi import FastAPI

app = FastAPI()


@app.post("/train")
async def train():
    pass


@app.post("/predict")
async def predict():
    pass
