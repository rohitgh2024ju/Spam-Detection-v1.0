# setup of api server
from fastapi import FastAPI, HTTPException, status, Form
from src.predict import predict_mail

app = FastAPI(
    title="Email Spam Detection API",
    description="Production-ready API for classifying emails using an SVM model. Accepts raw multi-line string inputs.",
    version="1.0",
)


@app.get("/", status_code=status.HTTP_200_OK)
def home():
    return {
        "status": "online",
        "model": "Linear SVM + TF-IDF",
        "dataset_scale": "83k+ rows",
        "documentation": "https://github.com/rohitgh2024ju/Spam-Detection-v1.0",
        "message": "Welcome! Send a POST request to /predict via Form Data to analyze an email.",
        "author": "Rohit Saha",
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_mail_trigger(message: str = Form(...)):
    """
    Takes text data in any format (including raw newlines, tabs, and unescaped quotes)
    via application/x-www-form-urlencoded or multipart/form-data.
    """
    if not message or not message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email message cannot be empty or blank whitespace.",
        )

    try:
        result = predict_mail(message)

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
