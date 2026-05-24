# setup of api server
from fastapi import FastAPI, HTTPException, status, Form
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from src.predict import predict_mail
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def groq_reasoning(prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
You are an email spam analysis assistant. And an email is classified as spam.

Your task:
- Analyze the email content.
- Return 2-3 concise spam indicators.
- Each reason must contain 3-6 words only.
- Use short warning-style phrases.
- Be specific, not generic.
- Do not use full sentences.
- Do not hallucinate or invent information.
- Only use evidence from the email.
- Output must be valid JSON.
- Do not include markdown.

Good examples:
- "Suspicious external verification link"
- "Creates urgent payment pressure"
- "Generic employee greeting"

Bad examples:
- "Urgent tone"
- "This email creates urgency"
- "The sender looks suspicious"

""",
            },
            {
                "role": "user",
                "content": f"""
Email:
{prompt}

Return output in this format:

{{
    "reasons": [
        "reason 1",
        "reason 2",
        "reason 3"
    ]
}}
""",
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2,
    )

    result = chat_completion.choices[0].message.content

    result_json = json.loads(result)

    return result_json["reasons"]


app = FastAPI(
    title="Email Spam Detection API",
    description="Production-ready API for classifying emails using an SVM model. Accepts raw multi-line string inputs.",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mail.google.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
async def predict_mail_trigger(message: str = Form(...)):
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

        reasons = None
        if prediction == "spam":
            reasons = groq_reasoning(mail)

        return {
            "mail": mail,
            "prediction": prediction,
            "metrics": {
                "decision_boundary_distance": round(distance, 4),
                "confidence_tier": confidence,
            },
            "reasons": reasons,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction engine error: {str(e)}",
        )
