from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import io
import base64
from model import SubstanceDetectionModel  # Your model file

app = Flask(__name__)

# Load the trained model
model = SubstanceDetectionModel()
model.load_state_dict(torch.load('models/model.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image']
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Process the image, extract features, and get prediction
    # For simplicity, assume 'process_image' is a function that prepares the image for prediction
    image = process_image(image) 
    prediction = model(image)
    
    # Assume prediction returns a dictionary with substance info, hex, confidence, toxicity
    result = {
        'substance': prediction['substance'],
        'hex': prediction['hex'],
        'confidence': prediction['confidence'],
        'info': prediction['info'],
        'toxicity': prediction['toxicity']
    }
    return jsonify(result)

def process_image(image):
    # Process the image to match the input format of your model
    # Example: Convert to tensor, normalize, etc.
    return image

if __name__ == '__main__':
    app.run(debug=True)
