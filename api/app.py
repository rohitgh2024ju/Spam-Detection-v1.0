# setup of api server
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from src.predict import predict_mail

app = FastAPI(
    title="Email Spam Detection API",
    description="Production-ready API for classifying emails using an SVM model.",
    version="1.0",
)

class Email(BaseModel):
    message: str


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {
        "status": "online",
        "model": "Linear SVM + TF-IDF",
        "dataset_scale": "83k+ rows",
        "documentation": "https://github.com/rohitgh2024ju/Spam-Detection-v1.0",
        "message": "Welcome! Send a POST request to /predict with a JSON body to analyze an email.",
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_mail_trigger(email: Email):
    if not email.message or not email.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email message cannot be empty or blank whitespace.",
        )

    try:
        result = predict_mail(email.message)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Processing failed. Invalid input format.",
            )

        mail, prediction, distance, confidence = result

        return {
            "mail": mail,
            "prediction": prediction,
            "metrics": {
                "decision_boundary_distance": round(distance, 4),
                "confidence_tier": confidence,
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction engine error: {str(e)}",
        )
