import cv2
import numpy as np
import tensorflow as tf

# 1. Modeli Yükle (TFLite Interpreter)
interpreter = tf.lite.Interpreter(model_path="models/model.tflite")
interpreter.allocate_tensors()

# Giriş ve Çıkış detaylarını al
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Kamerayı Başlat
cap = cv2.VideoCapture(0)

print("Test başlıyor... Çıkmak için 'q' tuşuna bas.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Görüntü Ön İşleme (Eğitimdekiyle aynı olmalı)
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0  # Normalize et
    img = np.expand_dims(img, axis=0)     # (1, 224, 224, 3) formatına getir
    
    # 3. Modeli Çalıştır (Inference)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    # Sonucu Al
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    
    # 4. Ekrana Yazdır (Lamba Simülasyonu)
    # 0 (Berat) - 1 (Diger) mantığına göre (Sigmoid çıktısı)
    if prediction < 0.15:
        label = f"ERISIM ONAYLANDI (Guven: {100*(1-prediction):.1f}%)"
        color = (0, 255, 0) # Yeşil
    else:
        label = "ERISIM REDDEDILDI"
        color = (0, 0, 255) # Kırmızı
        
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Edge AI Test Paneli", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()