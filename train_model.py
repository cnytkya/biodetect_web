import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from utils.data_preprocessing import load_and_preprocess_data
import os
import optuna
from optuna.integration import TFKerasPruningCallback
import json # Sınıf indekslerini kaydetmek için

# Model parametreleri
IMG_HEIGHT = 128
IMG_WIDTH = 128
DATA_DIR = 'data/PlantVillage' # PlantVillage veri setinizin yolu
MODEL_SAVE_PATH = 'models/plant_classifier.h5'

# Optuna objective fonksiyonu için global sınıf sayısı
# Bu, objective fonksiyonu her çağrıldığında veri setini tekrar yüklememek için
# sınıf sayısını bir kez önceden belirlememizi sağlar.
GLOBAL_NUM_CLASSES = 0 
try:
    temp_train_generator, _ = load_and_preprocess_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, 32) # Geçici bir batch_size ile yükle
    GLOBAL_NUM_CLASSES = len(temp_train_generator.class_indices)
    print(f"Global Toplam sınıf sayısı: {GLOBAL_NUM_CLASSES}")
except Exception as e:
    print(f"Veri seti ön yüklemesinde hata oluştu: {e}")
    # GLOBAL_NUM_CLASSES hala 0 kalacak. Bu durumda aşağıdaki kodlar da sorun yaşayabilir.
    # Gerçek uygulamada burada bir loglama veya çıkış mekanizması eklenebilir.

def objective(trial):
    """
    Optuna tarafından optimize edilecek objektif fonksiyon.
    Bu fonksiyon, belirli bir hiperparametre kombinasyonu ile modeli eğitir
    ve doğrulama doğruluğunu döndürür.
    """
    # Hiperparametreleri Optuna'dan öneri olarak al
    epochs = trial.suggest_int('epochs', 10, 50, step=5) # 10 ile 50 arasında 5'er adım
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64]) # Bu değerlerden birini seç
    dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.7, step=0.1) # 0.3 ile 0.7 arasında 0.1'er adım
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2) # Logaritmik ölçekte öğrenme oranı
    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'rmsprop', 'sgd'])
    
    # Model mimarisi için hiperparametreler
    num_conv_layers = trial.suggest_int('num_conv_layers', 2, 4) # 2 ile 4 arasında katman sayısı
    filters_per_layer = trial.suggest_categorical('filters_per_layer', [32, 64, 128]) # Her katman için başlangıç filtre sayısı

    # Modeli oluştur
    model = Sequential()
    
    # İlk Conv katmanı
    model.add(Conv2D(filters_per_layer, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(MaxPooling2D((2, 2)))

    # Ek Conv katmanları
    for i in range(num_conv_layers - 1): # İlk katman zaten eklendiği için -1
        # Filtre sayısını her katmanda ikiye katla veya sabit tutmak isterseniz filters_per_layer kullanın.
        # Şu anki mantık, ilk katmanın filtre sayısını baz alıp artırıyor.
        model.add(Conv2D(filters_per_layer * (2**(i+1)), (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    # Dense katmandaki nöron sayısını optimize et
    model.add(Dense(trial.suggest_categorical('dense_units', [128, 256, 512]), activation='relu'))
    model.add(Dropout(dropout_rate)) # Dropout oranını optimize et
    model.add(Dense(GLOBAL_NUM_CLASSES, activation='softmax')) # Çıkış katmanı sınıf sayısı

    # Optimizatör seçimi
    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else: # sgd
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Veri jeneratörlerini batch_size'a göre yeniden oluştur
    # Çünkü batch_size da optimize edilen bir hiperparametre.
    train_generator, validation_generator = load_and_preprocess_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, batch_size)

    # Callbacks tanımlama (objective fonksiyonu içinde yerel)
    # Erken durdurma: Doğrulama kaybı belirli bir sabır süresi boyunca iyileşmezse eğitimi durdur.
    early_stopping_obj = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Optuna'nın erken durdurma (pruning) callback'i
    pruning_callback = TFKerasPruningCallback(trial, 'val_accuracy') # val_accuracy'ye göre budama yap

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size, # Tam batch sayısını kullan (son batch eksikse uyarı verebilir)
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size, # Tam batch sayısını kullan
        callbacks=[early_stopping_obj, pruning_callback], # Callbacks'i ekle
        verbose=0 # Optuna denemeleri sırasında eğitimin detaylı çıktısını gösterme
    )

    # Optuna'nın optimize edeceği değeri döndür (genellikle doğrulama doğruluğu)
    return history.history['val_accuracy'][-1] # Son epoch'un doğrulama doğruluğunu döndür

if __name__ == '__main__':
    # Modelleri kaydetmek için 'models' klasörünün var olduğundan emin ol
    os.makedirs('models', exist_ok=True)

    # Sınıf indekslerini kaydet (Optuna optimizasyonundan önce yapılır ve tek seferliktir)
    class_indices_path = 'models/class_indices.json'
    # GLOBAL_NUM_CLASSES'ı belirlerken kullanılan temp_train_generator'ı kontrol et
    # Bu kısmın doğru çalışması için temp_train_generator'ın var olması ve sınıf_indices'in dolu olması gerekir.
    if GLOBAL_NUM_CLASSES > 0 and 'temp_train_generator' in globals(): # temp_train_generator'ın globalde varlığını kontrol et
        sorted_class_names = [name for name, index in sorted(temp_train_generator.class_indices.items(), key=lambda item: item[1])]
        with open(class_indices_path, 'w') as f:
            json.dump(sorted_class_names, f)
        print(f"Sınıf isimleri başarıyla kaydedildi: {class_indices_path}")
    else:
        print("Sınıf isimleri kaydedilemedi çünkü veri seti yüklenemedi veya sınıf sayısı sıfır. Lütfen DATA_DIR yolunu kontrol edin.")


    print("Optuna ile hiperparametre optimizasyonu başlatılıyor...")
    # Optuna study oluştur
    # 'maximize' yönü, 'val_accuracy' gibi daha yüksek değerlerin daha iyi olduğunu gösterir.
    study = optuna.create_study(direction='maximize')

    # Optimizasyon sürecini başlat
    # n_trials: Kaç farklı hiperparametre kombinasyonunu deneyecek.
    # timeout: Toplam optimizasyon için maksimum süre (saniye).
    # callbacks: Optimizasyon sırasında çalışacak callback'ler (örn. ilerlemeyi bildirme).
    study.optimize(objective, n_trials=50, timeout=600) # Örneğin 50 deneme veya 600 saniye (10 dakika)

    print("\nOptimizasyon tamamlandı.")
    print("En iyi deneme:")
    print(f"  Değer (Doğruluk): {study.best_trial.value:.4f}")
    print(f"  Parametreler: {study.best_trial.params}")

    # En iyi parametrelerle nihai modeli eğitme ve kaydetme
    print("\nEn iyi parametrelerle nihai model eğitiliyor ve kaydediliyor...")
    best_params = study.best_trial.params

    # En iyi parametreleri kullanarak nihai model mimarisini oluştur
    final_model = Sequential()
    
    # İlk Conv katmanı
    final_model.add(Conv2D(best_params['filters_per_layer'], (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
    final_model.add(MaxPooling2D((2, 2)))

    # Ek Conv katmanları
    for i in range(best_params['num_conv_layers'] - 1):
        final_model.add(Conv2D(best_params['filters_per_layer'] * (2**(i+1)), (3, 3), activation='relu'))
        final_model.add(MaxPooling2D((2, 2)))
    
    final_model.add(Flatten())
    final_model.add(Dense(best_params['dense_units'], activation='relu'))
    final_model.add(Dropout(best_params['dropout_rate']))
    final_model.add(Dense(GLOBAL_NUM_CLASSES, activation='softmax'))

    # Optimizatör seçimi (en iyi parametrelerden)
    final_optimizer_name = best_params['optimizer']
    final_learning_rate = best_params['learning_rate']

    if final_optimizer_name == 'adam':
        final_optimizer = tf.keras.optimizers.Adam(learning_rate=final_learning_rate)
    elif final_optimizer_name == 'rmsprop':
        final_optimizer = tf.keras.optimizers.RMSprop(learning_rate=final_learning_rate)
    else: # sgd
        final_optimizer = tf.keras.optimizers.SGD(learning_rate=final_learning_rate)

    final_model.compile(optimizer=final_optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    # Nihai eğitim için veri jeneratörlerini oluştur (en iyi batch_size ile)
    train_generator_final, validation_generator_final = load_and_preprocess_data(DATA_DIR, IMG_HEIGHT, IMG_WIDTH, best_params['batch_size'])

    # Nihai modeli kaydetmek için ModelCheckpoint'i kullanın (en iyi doğrulukta kaydedecek)
    final_model_checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

    # Nihai eğitim için EarlyStopping callback'i (Optuna objective içindeki ile ayrı bir nesne)
    early_stopping_final = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True) # Daha fazla sabır verebiliriz

    # Nihai modeli eğit
    final_model.fit(
        train_generator_final,
        steps_per_epoch=train_generator_final.samples // best_params['batch_size'],
        epochs=best_params['epochs'], # Optuna'nın bulduğu en iyi epoch sayısını kullanabiliriz
        validation_data=validation_generator_final,
        validation_steps=validation_generator_final.samples // best_params['batch_size'],
        callbacks=[early_stopping_final, final_model_checkpoint], # Callbacks'i ekle
        verbose=1 # Nihai eğitim çıktısını göster
    )

    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Nihai model başarıyla kaydedildi: {MODEL_SAVE_PATH}")
    else:
        print(f"Nihai model kaydedilemedi: {MODEL_SAVE_PATH}")