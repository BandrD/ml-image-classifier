from flask import Flask, request, jsonify, render_template_string
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Новый обработчик для главной страницы
@app.route('/')
def home():
    return """
    <h1>ML Image Classifier</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Predict</button>
    </form>
    <p>Или отправьте POST-запрос с изображением на /predict</p>
    """

# Обработчик для предсказаний
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img = Image.open(request.files['image']).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
    result = [{"class": label, "probability": float(prob)} for (_, label, prob) in decoded]
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
