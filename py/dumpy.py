# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
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

# Features and target
X = df.drop(columns=["label"])
y = df["label"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Model Initialization
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Backward Feature Selection
bfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='backward', cv=5, n_jobs=-1)
bfs.fit(X_train, y_train)

# Selected important features
selected_features = X.columns[bfs.get_support()].tolist()
print("✅ Selected Features after Backward Selection:", selected_features)

# 7. Train Model on Selected Features
X_train_selected = X_train[:, bfs.get_support()]
X_test_selected = X_test[:, bfs.get_support()]

model.fit(X_train_selected, y_train)

# 8. Model Evaluation
y_pred = model.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Accuracy with Selected Features:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=1))

# 9. Save the Final Model, Encoder, Scaler, and Selected Features
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

print("✅ Model, Encoder, Scaler, and Selected Features saved successfully!")

# 10. Sample Prediction (Optional)
sample = [[90, 42, 43, 20.5, 82.0, 6.5, 200]]
sample_scaled = scaler.transform(sample)
sample_selected = sample_scaled[:, bfs.get_support()]
predicted_crop = encoder.inverse_transform(model.predict(sample_selected))

print("✅ Recommended Crop for Sample:", predicted_crop[0])
