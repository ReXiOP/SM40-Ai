from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

YOLOv5_WEIGHTS = 'yolov5s.pt'
YOLOv5_IMG_SIZE = 640

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def anonymize_face_pixelate(image, blocks=20):
    (h, w) = image.shape[:2]
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")

    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]

            roi = image[start_y:end_y, start_x:end_x]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            pixelated_image = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (B, G, R), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = ""
            font_scale = 8
            font_thickness = 15

            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

            org = ((pixelated_image.shape[1] - text_size[0]) // 2, (pixelated_image.shape[0] + text_size[1]) // 2)
            font_color = (0, 0, 255)  # BGR color format

            cv2.putText(pixelated_image, text, org, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return pixelated_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        pixelated_image = anonymize_face_pixelate(image)

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pixelated_' + filename)
        cv2.imwrite(output_path, pixelated_image)

        return render_template('index.html', original_image=filename, pixelated_image='pixelated_' + filename)

    return render_template('index.html', error='Invalid file format')

@app.route('/download/<filename>')
def download(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
