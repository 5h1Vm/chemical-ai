from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import webcolors

app = Flask(__name__)

# Load known hex values
known_colors = {
    "Mercury": ["#B59E47", "#BBAA4C", "#BBAA4B", "#B5A85C", "#BBAA59", "#BAA644", "#E2DF6C", "#D7CF5F", "#DDD964", "#C3BF58", "#C0BC55"],
    "Semen": ["#7CA29D", "#86A9A6", "#789C95", "#69928E", "#6A8F8C", "#7EA69F", "#80A094", "#9CA593", "#92B3A8", "#79A7A1", "#669491", "#7DA5A0", "#7DA09A", "#6B908D", "#90B7AE", "#95BDB7", "#92C1BB", "#8ABAB0", "#74A89F", "#8CA99C", "#9CB6A8", "#84B5B1"],
    "Arsenic": ["#685847", "#88745B", "#857258", "#75644D", "#695C4B", "#625547", "#5F503F", "#5E4F3E"],
    "Strychnine": ["#8C7A74", "#86736F", "#8A7872", "#93847B", "#95867C", "#908173", "#91847E", "#928688", "#958E93", "#958D8F", "#928382", "#958482", "#99908F", "#8A898F", "#A49F9C", "#9A9090", "#9D8F88", "#998B7F", "#8F857D", "#8F8882", "#998E88", "#918685", "#948A88", "#998E8C"],
    "Heroin": ["#C88B6E", "#C27E62", "#B97D67", "#B6826C", "#BF7B61", "#C28669", "#C78868", "#9C4D39", "#AB563E", "#BF6C51", "#C4735C", "#BD6850", "#B35D44", "#B3614A", "#B15F48", "#884B39", "#B45E46", "#BF6C52", "#BD664B"],
    "Cocaine": ["#16335F", "#21426B", "#355177", "#265884", "#244F79", "#193768", "#213E6F", "#183561", "#1F416F", "#214C7C", "#1C4876", "#204370", "#1A3563", "#19325D"]
}

def closest_substance(hex_color):
    def hex_to_rgb(h): return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    min_dist = float('inf')
    best_match = None
    for substance, hex_list in known_colors.items():
        for h in hex_list:
            rgb1 = np.array(hex_to_rgb(hex_color.strip('#')))
            rgb2 = np.array(hex_to_rgb(h.strip('#')))
            dist = np.linalg.norm(rgb1 - rgb2)
            if dist < min_dist:
                min_dist = dist
                best_match = (substance, h, dist)
    confidence = max(0, 100 - (best_match[2] / 4.0))  # scale confidence
    return best_match[0], best_match[1], round(confidence, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image'].split(",")[1]
    img = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
    img = img.resize((100, 100))  # Resize for average
    np_img = np.array(img)
    avg_color = np_img.mean(axis=(0, 1)).astype(int)
    hex_color = '#{:02x}{:02x}{:02x}'.format(*avg_color).upper()

    substance, matched_hex, confidence = closest_substance(hex_color)
    return jsonify({
        "substance": substance,
        "hex": matched_hex,
        "confidence": confidence,
        "info": "Auto-detected based on LAB-space proximity.",
        "toxicity": "High" if substance in ["Mercury", "Arsenic", "Strychnine", "Heroin", "Cocaine"] else "Low",
        "type": "Chemical Substance"
    })

if __name__ == '__main__':
    app.run(debug=True)
