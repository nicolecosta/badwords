import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
import numpy as np
import pandas as pd
import uvicorn

app = FastAPI()

data = pd.read_csv('/home/nicole.sarvasi/Projetos/badwords/app/dataset/compileddata.csv')

# Load the model
loaded_model = joblib.load('/home/nicole.sarvasi/Projetos/badwords/app/badnames_model.joblib')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['words'])

@app.post("/classifier/")
async def inference(input_text: str):

    input_embedding = vectorizer.transform([input_text.lower()])

    # Use the trained model to obtain the anomaly score for the input text
    score = loaded_model.decision_function(input_embedding)

    # Set a threshold to determine if the input text is an anomaly or not
    threshold = 350

    # Compare the anomaly score with the threshold to determine the result
    if score < threshold:
        result = True
    else:
        result = False

    return {"result": result}

