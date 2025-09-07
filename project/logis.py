# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 2. Load the Dataset
df = pd.read_csv("Crop_recommendation.csv")

# 3. Data Preprocessing
print("Dataset Shape:", df.shape)
print(df.head())

# Encode target labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Features and Target
X = df.drop(columns=["label"])
y = df["label"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Initialization
model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs', multi_class='multinomial')

# 6. Train the Model
model.fit(X_train, y_train)

# 7. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Logistic Regression Model Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# 8. Save the Model, Scaler, and Encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Logistic Regression Model, Encoder, and Scaler saved successfully!")

# 9. Sample Prediction (Optional)
sample = [[90, 42, 43, 20.5, 82.0, 6.5, 200]]
sample_scaled = scaler.transform(sample)
predicted_crop = encoder.inverse_transform(model.predict(sample_scaled))

print("✅ Recommended Crop for Sample:", predicted_crop[0])
