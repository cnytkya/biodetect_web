import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_data(data_dir, img_height, img_width, batch_size, validation_split=0.2):
    """
    Veri setini yükler ve önişleme yapar.

    Args:
        data_dir (str): Veri setinin ana dizini (örn: 'data/PlantVillage').
        img_height (int): Görüntü yüksekliği.
        img_width (int): Görüntü genişliği.
        batch_size (int): Batch boyutu.
        validation_split (float): Doğrulama seti için ayrılacak oran.

    Returns:
        tuple: Eğitim ve doğrulama veri jeneratörleri.
    """
    # Veri artırma ve önişleme için ImageDataGenerator'ı tanımla
    train_datagen = ImageDataGenerator(
        rescale=1./255,                 # Piksel değerlerini 0-1 arasına ölçekle
        rotation_range=20,              # Rastgele döndürme
        width_shift_range=0.2,          # Rastgele yatay kaydırma
        height_shift_range=0.2,         # Rastgele dikey kaydırma
        shear_range=0.2,                # Makaslama dönüşümü
        zoom_range=0.2,                 # Rastgele yakınlaştırma
        horizontal_flip=True,           # Rastgele yatay çevirme
        fill_mode='nearest',            # Boş pikselleri doldurma yöntemi
        validation_split=validation_split # Doğrulama seti için ayrılacak oran
    )

    # Doğrulama seti için sadece yeniden ölçekleme
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    # Eğitim veri jeneratörünü oluştur
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',       # Sınıflandırma için kategoriye uygun etiketler
        subset='training'               # Eğitim seti
    )

    # Doğrulama veri jeneratörünü oluştur
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'             # Doğrulama seti
    )

    return train_generator, validation_generator

if __name__ == '__main__':
    # Test amaçlı kullanım
    data_dir = '../data/PlantVillage' #veri seti yolu 
    img_height = 128
    img_width = 128
    batch_size = 32

    try:
        train_gen, val_gen = load_and_preprocess_data(data_dir, img_height, img_width, batch_size)
        print(f"Eğitim veri seti bulundu: {train_gen.samples} örnek")
        print(f"Doğrulama veri seti bulundu: {val_gen.samples} örnek")
        print(f"Sınıf isimleri: {train_gen.class_indices}")
    except Exception as e:
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        print("Lütfen 'data/PlantVillage' klasörünün doğru konumda ve resimlerin alt klasörlerde olduğundan emin olun.")