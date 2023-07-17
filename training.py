import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
import joblib
from sklearn.metrics import accuracy_score

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Read the Excel file into a DataFrame
df = pd.read_excel('/home/ubuntu/Projects/Labs/badwords+names/badwords+names.xlsx').dropna()

# Define a function to apply the embedding model to each text instance
def embed_text(text):
    embeddings = embed([text])  # Apply the embedding model to a single text instance
    return embeddings.numpy()[0]  # Return the embedding as a numpy array

# Apply the embedding function to the 'text' column of the DataFrame
df['embeddings'] = df['words'].apply(embed_text)

# Encoding class labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Creating the One-Class SVM model
model = OneClassSVM()

# Splitting the data into features (X) and class (y)
X = df['embeddings'].to_list()
y = df['class']

# Training the model
model.fit(X)

# Getting the decision function scores
scores = model.decision_function(X)

# Checking if scores are above a threshold (e.g., 0) to determine abnormal instances
threshold = 0.0
is_abnormal = scores < threshold

# Adding the abnormality flag to the DataFrame
df['is_abnormal'] = is_abnormal

# Printing instances with low scores (not considered as anomalies)
low_score_instances = df[df['is_abnormal'] == False]
print("Instances with low scores:")
# Assuming you have a DataFrame called 'labels' with the ground truth labels
ground_truth = df['is_abnormal']  # Ground truth abnormality labels

# Calculate accuracy
accuracy = accuracy_score(ground_truth, is_abnormal)
print("Accuracy:", accuracy)
print(low_score_instances)

# Save the model
joblib.dump(model, '/home/ubuntu/Projects/Labs/badwords+names/badnames_model.joblib')



