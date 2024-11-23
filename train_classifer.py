import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Paths to features and labels
features_path = 'features/features.npy'
labels_path = 'features/labels.npy'
model_path = 'features/svm_classifier.pkl'

# Load features and labels
features = np.load(features_path)
labels = np.load(labels_path)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)

# Test the classifier
y_pred = svm_classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained classifier
with open(model_path, 'wb') as f:
    pickle.dump(svm_classifier, f)

print("Model training completed and saved successfully!")



