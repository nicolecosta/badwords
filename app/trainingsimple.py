# %%
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.metrics import accuracy_score


df_leet = pd.read_csv('/home/nicole.sarvasi/Projetos/badwords/app/dataset/leetdataset.csv', header=None, names=["0"])

# Split the values in the single column by comma and explode into separate rows
df_leet = df_leet["0"].str.split(",", expand=True).stack().reset_index(drop=True)

# Convert the resulting series into a DataFrame
df_leet = pd.DataFrame(df_leet, columns=["words"])

# Add a new column called "class" with all values set to "badword"
df_leet["class"] = "badword"

# Read the Excel file into a DataFrame
df_clean = pd.read_excel('/home/nicole.sarvasi/Projetos/badwords/app/dataset/badwords+names.xlsx').dropna()

df = pd.concat([df_clean, df_leet])

# Encoding class labels
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['words'])

# Creating the One-Class SVM model
model = OneClassSVM()

# Splitting the data into features (X) and class (y)
y = df['class']

# Training the model
model.fit(X)

# %%

# Getting the decision function scores
scores = model.decision_function(X)

# Checking if scores are above a threshold (e.g., 0) to determine abnormal instances
threshold = 300
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
joblib.dump(model, '/home/nicole.sarvasi/Projetos/badwords/app/badnames_model.joblib')
loaded_model = joblib.load('/home/nicole.sarvasi/Projetos/badwords/app/badnames_model.joblib')
# %%
# Transform the new text using the same vectorizer
new_text_vectorized = vectorizer.transform(['lazarento'])

# Predict whether the new text is an outlier (1) or not (0)
prediction = loaded_model.decision_function(new_text_vectorized)
print(prediction)



# %%
