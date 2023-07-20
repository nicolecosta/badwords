import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
from fastapi import FastAPI
import numpy as np
import uvicorn

app = FastAPI()

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Load the model
loaded_model = joblib.load('/home/ubuntu/Projects/Labs/badwords+names/app/badnames_model.joblib')

@app.post("/classifier/")
async def inference(input_text: str):

    white_list = ['ze', 'zé', 'jose', 'joao', 'josé', 'joão']

    if input_text.lower() not in white_list:
        input_embedding = embed([input_text.lower()])[0].numpy()

        # Use the trained model to obtain the anomaly score for the input text
        score = loaded_model.decision_function([input_embedding])

        # Set a threshold to determine if the input text is an anomaly or not
        threshold = -35

        # Compare the anomaly score with the threshold to determine the result
        if score < threshold:
            result = False
        else:
            result = True
    else:
        result = False
        #score = [0]

    return {"result": result}
