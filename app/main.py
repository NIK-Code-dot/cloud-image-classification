from PIL import Image as PILImage
from fastapi.middleware.cors import CORSMiddleware
from app.ml_model import predict_image
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session

from app.services.database import engine, Base, get_db
from app.models import Image, Prediction
from app.schemas import ImageCreate, ImageResponse

app = FastAPI(
    title="Image Classification API",
    description="Cloud-based Image Classification Project",
    version="1.0.0"
)
Base.metadata.create_all(bind=engine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "https://nik-code-dot.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "Image Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/images", response_model=ImageResponse)
def create_image(
    image: ImageCreate,
    db: Session = Depends(get_db)
):
    new_image = Image(
        filename=image.filename,
        file_path=image.file_path
    )

    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    return new_image
@app.get("/images", response_model=list[ImageResponse])
def get_images(db: Session = Depends(get_db)):
    images = db.query(Image).all()
    return images
from fastapi import UploadFile, File
import os
import shutil

UPLOAD_DIR = "ml/data/raw"

@app.post("/upload-image")
def upload_image(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image uploaded successfully",
        "filename": file.filename,
        "file_path": file_path
    }
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    image = PILImage.open(file.file).convert("RGB")

    predicted_class, confidence = predict_image(image)

    new_image = Image(
        filename=file.filename,
        file_path=None
    )

    db.add(new_image)
    db.commit()
    db.refresh(new_image)

    prediction = Prediction(
        image_id=new_image.id,
        predicted_class=predicted_class,
        confidence=confidence
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "filename": file.filename,
        "image_id": new_image.id,
        "prediction_id": prediction.id,
        "predicted_class": predicted_class,
        "confidence": confidence
    }

