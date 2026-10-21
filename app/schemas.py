from pydantic import BaseModel
from datetime import datetime


class ImageCreate(BaseModel):
    filename: str
    file_path: str | None = None


class ImageResponse(BaseModel):
    id: int
    filename: str
    file_path: str | None
    uploaded_at: datetime | None

    class Config:
        from_attributes = True
