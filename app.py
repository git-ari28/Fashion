import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import pickle
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'dataset'

# Load pre-trained models
category_classifier = pickle.load(open('models/category_classifier.pkl', 'rb'))
label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

# Ensure uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to extract features from images using ResNet50
def extract_features(image_path):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()  # Flatten the feature vector

# Route: Gender Selection
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        gender = request.form.get('gender')
        if not gender:
            return "Error: Gender is missing."
        return redirect(url_for('upload', gender=gender))
    return render_template('index.html')

# Route: Upload Image
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    gender = request.args.get('gender')
    if not gender:
        return "Error: Gender is missing."

    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part"
        file = request.files['image']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Extract features and classify category (topwear or bottomwear)
            uploaded_features = extract_features(filepath)
            predicted_category_index = category_classifier.predict([uploaded_features])[0]
            predicted_category = label_encoder.inverse_transform([predicted_category_index])[0]

            return redirect(url_for('preferences', gender=gender, category=predicted_category))
    return render_template('upload.html', gender=gender)

# Route: Choose Preferences from Other Categories
@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    gender = request.args.get('gender')
    category = request.args.get('category')

    if not gender or not category:
        return "Error: Gender or category is missing."

    # Define other categories to choose preferences from
    other_categories = ['bottomwear', 'shoes', 'accessories']
    if category in other_categories:
        other_categories.remove(category)

    # Prepare to display subcategories for the other categories
    subcategory_folders = {}
    for cat in other_categories:
        category_folder_path = os.path.join(app.config['DATASET_FOLDER'], gender, cat)
        if os.path.exists(category_folder_path):
            subcategory_folders[cat] = os.listdir(category_folder_path)

    if request.method == 'POST':
        preferences = {}
        for cat in other_categories:
            preference = request.form.get(f'{cat}_preference')
            if not preference:
                return f"Error: Preference for {cat} is missing."
            preferences[cat] = preference
        
        # Redirect to display images from the selected preferences
        return redirect(url_for('results', gender=gender, category=category, preferences=preferences))

    return render_template('preferences.html', gender=gender, category=category, subcategory_folders=subcategory_folders)

# Route: Show Images from Selected Preferences
@app.route('/results', methods=['GET'])
def results():
    gender = request.args.get('gender')
    category = request.args.get('category')
    preferences = request.args.get('preferences')

    if not gender or not category or not preferences:
        return "Error: Missing gender, category, or preferences."

    # Convert preferences from string to dict
    preferences = eval(preferences)

    images = {}
    for cat, preference in preferences.items():
        preference_folder = os.path.join(app.config['DATASET_FOLDER'], gender, cat, preference)
        if not os.path.exists(preference_folder):
            return f"Error: {preference} not found in {cat} subcategory."
        images[cat] = os.listdir(preference_folder)[:3]  # Get top 3 images

    # Prepare paths for images to display
    image_paths = {}
    for cat, imgs in images.items():
        image_paths[cat] = [os.path.join(gender, cat, preference, img) for img in imgs]

    return render_template('result.html', category=category, preferences=preferences, image_paths=image_paths)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
















