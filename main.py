from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from skimage import io
from skimage.filters import threshold_sauvola, threshold_niblack
import numpy as np
import os
import cv2
from PIL import Image, ImageEnhance
from skimage.util import img_as_uint, img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('uploaded_file', filename=filename))


@app.route('/uploaded_file/<filename>')
def uploaded_file(filename):
    return render_template('process.html', filename=filename)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/apply_filter/<filename>')
def apply_filter(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = Image.open(file_path)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(7)

    processed_filename = f"processed_{filename}"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    img.save(processed_path)

    return render_template('result.html', original=filename, processed=processed_filename, method = "Увеличение резкости")



@app.route('/erosion_image/<filename>')
def erosion_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = io.imread(file_path, as_gray = True)
    selem = np.ones((5, 5))  # Структурный элемент 5x5
    eroded_img = erosion(img, selem)
    eroded_filename = f"eroded_{filename}"
    eroded_path = os.path.join(app.config['UPLOAD_FOLDER'], eroded_filename)
    cv2.imwrite(eroded_path, img_as_ubyte(eroded_img))
    return render_template('result.html', original=filename, processed=eroded_filename, method="Эрозия")


@app.route('/dilation_image/<filename>')
def dilation_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = io.imread(file_path, as_gray = True)
    selem = np.ones((5, 5))  # Структурный элемент 5x5
    dilated_img = dilation(img, selem)
    dilated_filename = f"dilated_{filename}"
    dilated_path = os.path.join(app.config['UPLOAD_FOLDER'], dilated_filename)
    cv2.imwrite(dilated_path, img_as_ubyte(dilated_img))
    return render_template('result.html', original=filename, processed=dilated_filename, method="Дилатация")


@app.route('/opening_image/<filename>')
def opening_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = io.imread(file_path, as_gray = True)
    selem = np.ones((5, 5))  # Структурный элемент 5x5
    opened_img = opening(img, selem)
    opened_filename = f"opened_{filename}"
    opened_path = os.path.join(app.config['UPLOAD_FOLDER'], opened_filename)
    cv2.imwrite(opened_path, img_as_ubyte(opened_img))
    return render_template('result.html', original=filename, processed=opened_filename, method="Размыкание")


@app.route('/closing_image/<filename>')
def closing_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = io.imread(file_path, as_gray = True)
    selem = np.ones((5, 5))  # Структурный элемент 5x5
    closed_img = closing(img, selem)
    closed_filename = f"closed_{filename}"
    closed_path = os.path.join(app.config['UPLOAD_FOLDER'], closed_filename)
    cv2.imwrite(closed_path, img_as_ubyte(closed_img))
    return render_template('result.html', original=filename, processed=closed_filename, method="Замыкание")


if __name__ == '__main__':
    app.run(debug=True)
