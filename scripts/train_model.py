import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 1. Ayarlar
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data"

# 2. Veri Hazırlama (Aynı kalıyor, çok iyi)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 3. Model Mimarisi (GÜNCELLENDİ)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), 
                                               include_top=False, 
                                               weights='imagenet')

# MÜHENDİSLİK DOKUNUŞU: Fine-Tuning
# Modelin tamamını değil, sadece son katmanlarını eğitime açıyoruz.
base_model.trainable = True
for layer in base_model.layers[:-30]:  # İlk katmanlar donsun, son 30 katman senin yüzünü öğrensin
    layer.trainable = False

# Kapasite artırıldı ve Aşırı Öğrenmeyi (Overfitting) engellemek için Dropout eklendi
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5), # Nöronların yarısını rastgele kapat (Ezberlemeyi önler)
    layers.Dense(128, activation='relu'), # Ara karar mekanizması eklendi
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid') # 0 (Berat) veya 1 (Diger)
])

# ÖNEMLİ: Fine-Tuning yaparken öğrenme oranını (learning_rate) çok düşük tutmalıyız (1e-4).
# Yoksa MobileNet'in önceden öğrendiği iyi bilgileri yok ederiz.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# 4. Eğitimi Başlat (GÜNCELLENDİ)
# Modelin ezberlemesini engellemek ve en iyi modeli yakalamak için "Erken Durdurma" ekliyoruz.
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\nEğitim başlıyor... (Fine-Tuning aktif)")
# Epoch sayısını artırdık, erken durdurma gerekirse 15'ten önce kesecek.
model.fit(train_generator, 
          epochs=15, 
          validation_data=validation_generator,
          callbacks=[early_stop])

# 5. TFLite Dönüşümü (Aynı kalıyor)
print("\nModel TFLite formatına dönüştürülüyor...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Modeli kaydet
os.makedirs("models", exist_ok=True)
with open('models/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("\nBAŞARILI! Gelişmiş 'models/model.tflite' dosyası oluşturuldu.")