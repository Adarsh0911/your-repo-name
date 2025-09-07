
# 1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 2. Load Dataset
df = pd.read_csv("Crop_recommendation.csv")
print("📊 Dataset Shape:", df.shape)
print(df.head())

# 3. Data Overview
print("\n🧾 Dataset Info:")
print(df.info())

print("\n📈 Summary Statistics:")
print(df.describe())

print("\n🌾 Crop Distribution:")
print(df['label'].value_counts())

# 4. Visualize Feature Correlation
plt.figure(figsize=(10, 7))
sns.heatmap(df.drop('label', axis=1).corr(), annot=True, cmap='YlGnBu')
plt.title("Feature Correlation Heatmap")
plt.show()

# 5. Check for Nulls
print("\n🔍 Null Values in Dataset:")
print(df.isnull().sum())

# 6. Encode Crop Labels
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# 7. Feature and Target
X = df.drop('label', axis=1)
y = df['label']

# 8. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Train Random Forest
print("\n🌳 Training Random Forest...")
start = time.time()
model = RandomForestClassifier()
model.fit(X_train, y_train)
end = time.time()

# 10. Evaluate Model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"⏱ Training Time: {end - start:.2f} seconds")
print("\n🧪 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# 11. Save Final Model and Label Encoder
with open("crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

# 12. Predict for Sample Input
sample_input = [[90, 42, 43, 20.5, 82.0, 6.5, 200]]
predicted_crop = encoder.inverse_transform(model.predict(sample_input))
print("\n🌱 Recommended Crop for Input:", predicted_crop[0])