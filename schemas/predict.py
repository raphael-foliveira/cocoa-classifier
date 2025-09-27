from pydantic import BaseModel


class PredictRequest(BaseModel):
    image_path: str
    min_area: int
    max_area: int
    open_ksize: int
