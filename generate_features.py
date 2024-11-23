import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import pickle

# Paths
dataset_dir = 'dataset'
features_dir = 'features'
if not os.path.exists(features_dir):
    os.makedirs(features_dir)

# Initialize ResNet model (pre-trained on ImageNet)
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
image_size = (224, 224)

# Lists to store data
feature_vectors = []
labels = []

# Iterate through dataset directory
for gender in os.listdir(dataset_dir):  # male, female
    gender_path = os.path.join(dataset_dir, gender)
    if os.path.isdir(gender_path):
        for category in os.listdir(gender_path):  # bottomwear, topwear, etc.
            category_path = os.path.join(gender_path, category)
            if os.path.isdir(category_path):
                for subcategory in os.listdir(category_path):  # Bags, Watches, etc.
                    subcategory_path = os.path.join(category_path, subcategory)
                    if os.path.isdir(subcategory_path):
                        for img_file in os.listdir(subcategory_path):
                            img_path = os.path.join(subcategory_path, img_file)
                            try:
                                # Load and preprocess the image
                                image = load_img(img_path, target_size=image_size)
                                image = img_to_array(image)
                                image = preprocess_input(image)
                                # Extract features
                                features = model.predict(np.expand_dims(image, axis=0))[0]
                                feature_vectors.append(features)
                                labels.append(f"{gender}_{category}_{subcategory}")
                            except Exception as e:
                                print(f"Error processing {img_path}: {e}")

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save features and labels
np.save(os.path.join(features_dir, 'features.npy'), np.array(feature_vectors))
np.save(os.path.join(features_dir, 'labels.npy'), np.array(encoded_labels))

# Save label encoder for later use
with open(os.path.join(features_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(label_encoder, f)

print("Feature extraction completed and saved!")



