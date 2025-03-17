from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from io import BytesIO
import webbrowser
import os

# Character mapping
encode = {'_n':'-น','_m':'-ม','_o':'-อ',
          'K':'ก','KH':'ข','C':'ค','NG':'ง','J':'จ',
          'CH':'ฉ','NN':'ณ','D':'ด','T':'ต','N':'น',
          'B':'บ','PA':'ป','PH':'ผ','F':'ฝ','P':'พ',
          'M':'ม','Y':'ย','R':'ร','L':'ล','V':'ว','S':'ส',
          'H':'ห','HL':'หลฯ','OY':'อยฯ','A':'ะ','Aa':'ั',
          'AAA':'า','EI':'ิ','EE':'ี','EU':'ุ','EA':'เ','AI':'ใ',
          }

train_dir = "dataset/model/train/"   #ที่อยู่ของไฟล์ภาพที่จะเอาไป train
val_dir = "dataset/model/test/"     #ที่อยู่ของไฟล์ภาพที่จะเอาไป train
target_img_shape = (70,70)           # ปรับขนาดภาพ
nclass = 35                         #คลาสตัวอักษรที่มี

train_datagen = ImageDataGenerator()
train_set = train_datagen.flow_from_directory(train_dir,target_size=target_img_shape,batch_size=32,class_mode = 'categorical')

val_datagen = ImageDataGenerator()
val_set = val_datagen.flow_from_directory(val_dir,shuffle=False,target_size=target_img_shape,batch_size=32,class_mode = 'categorical')

labels =(train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())

# Load model
model = load_model('Model/OCR-lanna.h5')
#webbrowser.open('file://' + os.path.realpath('template/OCR-test.html'))

# Flask app setup
app = Flask(__name__)
CORS(app)

# ✅ เพิ่ม route สำหรับ `/`
@app.route("/", methods=["GET", "HEAD" ,"POST"])
def home():
    return jsonify({"message": "OCR API is running!"})

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
    MIN_AREA = 40 
    MAX_AREA = 8000 
    filtered_contours = [cnt for cnt in contours if 40 < cv2.contourArea(cnt) < 8000]
    
    count = 0
    obj_Y = []
    obj_b = []
    
    # OCR Prediction
    predicts = []
    predictions = []  # เก็บการทำนายทั้งหมด
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        object_img = thresh[y:y + h, x:x + w]
        if h < 10 or w < 20:
            continue
        contours1, _ = cv2.findContours(object_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [cv2.contourArea(cnt) for cnt in contours1]
        max_area = max(areas)
        mask = np.zeros_like(object_img)
        
        for i, contour in enumerate(contours1):
            if cv2.contourArea(contour) == max_area: 
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                
        # Process object image
        result = cv2.bitwise_and(object_img, object_img, mask=mask)      
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        result = cv2.resize(result,(70,70))
        img_array = img_to_array(result)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        #ทำนายอักษรจากภาพ
        predict = model.predict(img_array)
        pred_cls = labels[np.argmax(predict, axis=-1)[0]]
        
        # แปลงค่าความมั่นใจเป็น %
        confidence = np.max(predict) * 100  
        # แสดงผลลัพธ์ใน console
        print(f"Prediction: {encode[pred_cls]} ({confidence:.2f}%)") 
        
        # เก็บการทำนายของตัวอักษรทั้งหมด
        predictions.append({
            'character': encode.get(pred_cls, pred_cls),  # ใช้ encode ถ้ามีในดิกชันนารี
            'confidence': confidence
        })
    #    result = cv2.resize(cv2.cvtColor(object_img, cv2.COLOR_GRAY2BGR), (70, 70))
    #    img_array = preprocess_input(np.expand_dims(img_to_array(result), axis=0))
    #    predict = model.predict(img_array)
        #ทำนายอักษรจากภาพ
        #pred_cls = max(enumerate(predict[0]), key=lambda x: x[1])[0]  # Get class index
        #predicts.append(encode.get(pred_cls, str(pred_cls)))
        
        
        object_center = [x+(w//2), y+(h//2)]   # คำนวณตำแหน่งศูนย์กลางของตัวอักษร
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  #วาดกรอบสี่เหลี่ยมรอบตัวอักษรและแสดงผลลัพธ์บนภาพ
        
        cv2.putText(image, f"{pred_cls} {confidence:.1f}%", (x, y - 10),  
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
        cv2.circle(image, (object_center[0], object_center[1]), 2, (0, 0, 255), -1)

        obj_Y.append(y)   #เก็บตำแหน่ง
        obj_b.append(y+h)
        predicts.append([encode[pred_cls],object_center,x,x+w])
        count += 1
    
    under_line = int(sum(obj_b)/len(obj_b))+4
    line = int(sum(obj_Y)/len(obj_Y))-3
    cv2.line(image, (0,line), (5000,line), (0, 0, 255), 1)
    cv2.line(image, (0,under_line), (5000,under_line), (0, 0, 255), 1)

    top_vowel = []
    under_vowel = []
    alphabet = []
    for i in range(len(predicts)):
        if predicts[i][1][1] < line :
            if predicts[i][0] in ['ค','ง','ั','ิ','ี','เ','ก','ม','ป','ย','พ','ว']:
                top_vowel.append(predicts[i])
            continue

        if predicts[i][1][1] > under_line :
            under_vowel.append(predicts[i])
            continue
        
        else :
            alphabet.append(predicts[i])

    alphabet.sort(key = lambda x : x[1][0])
    top_vowel.sort(key = lambda x : x[1][0])
    under_vowel.sort(key = lambda x : x[1][0])
    text = ""
    # print(top_vowel)
    # print(alphabet)
    # print(under_vowel)

    for i in range(len(alphabet)):
        if alphabet[i][0] in ['ิ','ี'] :
            text += 'ค'
        else :
            text += alphabet[i][0]

        for j in range(len(top_vowel)):
            if alphabet[i][2] < top_vowel[j][2] < alphabet[i][3] or alphabet[i][2] < top_vowel[j][1][0] < alphabet[i][3] :
                if top_vowel[j][0] in ['ค','ง','เ','ก','ม','ป','ย','พ','ว'] :
                    text += 'ิ'
                else :
                    text += top_vowel[j][0]
                top_vowel[j][0] = ""

        for j in range(len(under_vowel)):
            if alphabet[i][2] < under_vowel[j][2]+5 < alphabet[i][3] or alphabet[i][2] < under_vowel[j][1][0] < alphabet[i][3] :
                print(under_vowel[j][0])
                text += under_vowel[j][0]+"ฯ"
                under_vowel[j][0] = ""

    if "-" in text:text = text.replace("-","")

    text_edit = ""
    for t in range(len(text)):
        text_edit += text[t]
        if t == 0 or t == len(text)-1: continue
        if text[t] in ['ั','ิ','ี','ุ']:
            if text[t+1] in ['ั','ิ','ี','ุ','เ','ใ','า','ะ']:
                text_edit += 'น'

        if text[t] in ['ั','า','ิ','ี','ุ']:
            if text[t-1] == ['ั']:
                text_edit += 'น'

    if "ฯ" in text_edit:text_edit = text_edit.replace("ฯ","")

    #return jsonify({'text_LN': text, 'text': text_edit})
    # แสดงค่า confidence 
    #return jsonify({'text_LN': text, 'text': text_edit, 'confidence': confidence})  # Return the confidence value here
    
    # ส่งผลลัพธ์เป็น JSON พร้อมการทำนายของทุกตัวอักษร
#    return jsonify({'text_LN': text, 'text': text_edit, 'predictions': predictions}) 
#   text = ''.join(predicts)
#    return jsonify({'text': text})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)