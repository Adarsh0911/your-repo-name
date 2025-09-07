import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Encode crop labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Features & target
X = df.drop('label', axis=1)
y = df['label']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Model and encoder saved successfully!")

import pickle

# Final model selection
final_model = RandomForestClassifier()
final_model.fit(X, y)

# Save the model
with open("crop_model.pkl", "wb") as model_file:
    pickle.dump(final_model, model_file)

# Save the label encoder
with open("label_encoder.pkl", "wb") as enc_file:
    pickle.dump(encoder, enc_file)

print("✅ Model and encoder saved successfully!")
