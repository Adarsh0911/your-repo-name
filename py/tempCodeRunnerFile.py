import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('Crop_recommendation.csv')

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Binarize labels for multiclass ROC
classes = y.unique()
y_bin = label_binarize(y, classes=classes)
n_classes = y_bin.shape[1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.3, random_state=42)

# Train Random Forest for each class
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Get probability estimates
y_score = clf.predict_proba(X_test)

# For RandomForest with multilabel, we need to stack class outputs
# Ensure y_score is a list of arrays (one per class)
if isinstance(y_score, list):
    y_score = np.array([score[:, 1] for score in y_score]).T

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 7))
colors = plt.cm.get_cmap('tab10', n_classes)

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"{classes[i]} (AUC = {roc_auc[i]:.2f})", color=colors(i))

plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest - Crop Recommendation')
plt.legend(loc='lower right')
plt.grid()
plt.show()
