from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

# Character mapping
encode = {
    '_n': '-น', '_m': '-ม', '_o': '-อ', 'K': 'ก', 'KH': 'ฃ', 'C': 'ค', 'NG': 'ง',
    'J': 'จ', 'CH': 'ฉ', 'NN': 'ณ', 'D': 'ด', 'T': 'ต', 'N': 'น', 'B': 'บ', 'PA': 'ป',
    'PH': 'ผ', 'F': 'ฝ', 'P': 'พ', 'M': 'ม', 'Y': 'ย', 'R': 'ร', 'L': 'ล', 'V': 'ว',
    'S': 'ส', 'H': 'ห', 'HL': 'หลฯ', 'OY': 'อยฯ', 'A': 'ะ', 'Aa': 'ั', 'AAA': 'า',
    'EI': 'ิ', 'EE': 'ี', 'EU': 'ุ', 'EA': 'เ', 'AI': 'ใ'
}

# Load model
model = load_model('Model/OCR-lanna.h5')

# Flask app setup
app = Flask(__name__)
CORS(app)

@app.route('/process_image', methods=['POST'])
def process_image():
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image uploaded'}), 400
    
    # Process image
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Resize and preprocess image
    new_height = 150 
    h, w = image.shape[:2]
    new_width = int((new_height / h) * w)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1] 
    thresh = cv2.bitwise_not(thresh)
    
    # Find contours and filter them
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if 40 < cv2.contourArea(cnt) < 8000]
    
    # OCR Prediction
    predicts = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        object_img = thresh[y:y + h, x:x + w]
        if h < 10 or w < 20:
            continue
        
        # Process object image
        result = cv2.resize(cv2.cvtColor(object_img, cv2.COLOR_GRAY2BGR), (70, 70))
        img_array = preprocess_input(np.expand_dims(img_to_array(result), axis=0))
        predict = model.predict(img_array)
        pred_cls = max(enumerate(predict[0]), key=lambda x: x[1])[0]  # Get class index
        predicts.append(encode.get(pred_cls, str(pred_cls)))
    
    text = ''.join(predicts)
    return jsonify({'text': text})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)