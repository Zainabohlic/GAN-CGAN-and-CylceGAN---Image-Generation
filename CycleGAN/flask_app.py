import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow_addons.layers import InstanceNormalization  # Import InstanceNormalization from Addons

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the saved CycleGAN models (G1: face to sketch, G2: sketch to face)
with tf.keras.utils.custom_object_scope({'InstanceNormalization': InstanceNormalization}):
    G1 = tf.keras.models.load_model('./saved_model/G1_model.h5')
    G2 = tf.keras.models.load_model('./saved_model/G2_model.h5')

# Define the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Resize and preprocess the input image
def preprocess_image(image_path, is_sketch=False):
    # Open the image file and resize it to 64x64
    img = Image.open(image_path).resize((64, 64))
    
    if is_sketch:
        # If it's a sketch (grayscale), convert it to grayscale
        img = img.convert('L')  # 'L' mode converts to grayscale (1 channel)
    
    # Convert the image to a NumPy array and normalize it to [0, 1]
    img_array = np.array(img) / 255.0
    
    # If it's grayscale, add the channel dimension (needed for model input)
    if len(img_array.shape) == 2:  # Grayscale image
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    
    # Expand dimensions to fit model input (1, 64, 64, 1) or (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Route to handle image generation requests
@app.route('/generate', methods=['POST'])
def generate_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if the image is a sketch or face (use form data to determine this)
        is_sketch = request.form.get('is_sketch', 'false').lower() == 'true'
        
        # Preprocess the uploaded image, pass the 'is_sketch' flag
        img_array = preprocess_image(filepath, is_sketch=is_sketch)
        
        # Pass the image to the appropriate model
        if is_sketch:  # Convert sketch to face (G2)
            generated_image = G2.predict(img_array)
        else:  # Convert face to sketch (G1)
            generated_image = G1.predict(img_array)
        
        # Post-process the generated image (denormalize and convert to uint8)
        generated_image = (generated_image[0] * 255).astype(np.uint8)
        
        # Check if the output image is grayscale or RGB
        if generated_image.shape[-1] == 1:  # Grayscale image
            generated_image = generated_image.squeeze(axis=-1)  # Remove last channel dimension
        
        # Save the generated image
        output_filename = 'output_' + filename
        output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        output_image = Image.fromarray(generated_image)
        output_image.save(output_filepath)
        
        return jsonify({"output_image_path": f"/uploads/{output_filename}"}), 200

    return jsonify({"error": "File type not allowed"}), 400

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
