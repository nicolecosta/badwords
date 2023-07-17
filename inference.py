import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Load the model
loaded_model = joblib.load('/home/ubuntu/Projects/Labs/badwords+names/badnames_model.joblib')

input_embedding = embed(['dessa'])[0].numpy()

# Use the trained model to obtain the anomaly score for the input text
anomaly_score = loaded_model.decision_function([input_embedding])

# Set a threshold to determine if the input text is an anomaly or not
threshold = 0.7

# Compare the anomaly score with the threshold to determine the result
if anomaly_score < threshold:
    print("Input text is not an anomaly.")
else:
    print("Input text is an anomaly.")
