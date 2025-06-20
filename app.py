# app.py
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
import numpy as np
import os
import io
from PIL import Image
import json # Sınıf isimlerini okumak için eklendi

app = Flask(__name__)

# Modelin ve sınıf isimlerinin yolları
MODEL_PATH = 'models/plant_classifier.h5'
# Görüntü boyutları (model eğitiminde kullanılanlarla aynı olmalı)
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES_PATH = 'models/class_indices.json' # Sınıf isimleri dosya yolu eklendi

# Modeli yükle
model = None # Başlangıçta None olarak ayarla
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model başarıyla yüklendi.")
except Exception as e:
    print(f"Model yüklenirken bir hata oluştu: {e}")
    model = None # Hata durumunda model None olarak ayarlanır

# Sınıf isimlerini yükle (class_indices.json dosyasından)
CLASS_NAMES = [] # Başlangıçta boş liste olarak ayarlandı
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        CLASS_NAMES = json.load(f)
    print(f"Sınıf isimleri başarıyla yüklendi: {CLASS_NAMES}")
except FileNotFoundError:
    print(f"Hata: Sınıf isimleri dosyası bulunamadı: {CLASS_NAMES_PATH}")
    CLASS_NAMES = [] # Dosya bulunamazsa veya yüklenemezse boş liste
except Exception as e:
    print(f"Sınıf isimleri yüklenirken hata oluştu: {e}")
    CLASS_NAMES = [] # Hata durumunda boş liste


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model yüklenemedi. Lütfen sunucuyu kontrol edin.'})

    if not CLASS_NAMES: # Sınıf isimleri yüklenemediyse hata döndür
        return jsonify({'error': 'Sınıf isimleri yüklenemedi. Lütfen train_model.py\'nin başarılı bir şekilde çalışıp class_indices.json dosyasını oluşturduğundan emin olun.'})

    if 'file' not in request.files:
        return jsonify({'error': 'Dosya bulunamadı.'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi.'})

    if file:
        try:
            # Resmi yükle ve önişle
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Batch boyutu ekle
            img_array = img_array / 255.0 # Piksel değerlerini ölçekle

            # Tahmin yap
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            
            # Tahmin edilen indeksin CLASS_NAMES listesi içinde olduğundan emin olamamız gerekir. burayı tekrar kontrol edeceğim
            if predicted_class_index < len(CLASS_NAMES):
                predicted_class_name = CLASS_NAMES[predicted_class_index]
            else:
                predicted_class_name = "Bilinmeyen Sınıf (Indeks dışı veya sınıflar yüklenemedi)" # Güvenlik kontrolü

            confidence = float(np.max(predictions[0]))

            return jsonify({
                'predicted_class': predicted_class_name,
                'confidence': f'{confidence*100:.2f}%'
            })
        except Exception as e:
            return jsonify({'error': f'Tahmin sırasında bir hata oluştu: {e}'})

if __name__ == '__main__':
    # Flask uygulamasını başlat
    app.run(debug=True) # debug=True geliştirme aşamasında hataları görmek için iyidir.