# 🛡️ Edge AI Security System: Real-Time Face Recognition with C++ & TFLite

Bu proje, yüksek performanslı C++ motoru ve optimize edilmiş bir Derin Öğrenme (Deep Learning) modelini birleştiren uçtan uca bir **Edge AI** güvenlik sistemidir. Sistem, internet bağlantısına ihtiyaç duymadan (on-device) gerçek zamanlı yüz tanıma ve yetkilendirme işlemi gerçekleştirmektedir.

## 🚀 Öne Çıkan Özellikler
- **Yüksek Performans:** C++ ve OpenCV kullanılarak geliştirilen düşük gecikmeli (low-latency) çıkarım motoru.
- **Optimize Beyin:** MobileNetV2 mimarisi üzerine inşa edilmiş ve TensorFlow Lite (.tflite) formatına dönüştürülmüş hafifletilmiş model.
- **Fine-Tuning:** "Hard Negative Mining" teknikleri ile eğitilmiş, yanlış pozitifleri (False Positive) minimize eden özelleştirilmiş model katmanları.
- **Güvenlik Odaklı:** Ayarlanabilir eşik değeri (threshold) ile %95+ doğruluk oranı.

## 🛠️ Teknik Altyapı (Tech Stack)
- **Dil:** C++, Python
- **Kütüphaneler:** OpenCV (DNN Module), TensorFlow, Keras
- **Model Mimarisi:** MobileNetV2 (Pre-trained on ImageNet)
- **Format:** TFLite (TensorFlow Lite)
- **Derleme Sistemi:** CMake



## 📁 Proje Yapısı
```text
├── src/
│   ├── main.cpp             # C++ Gerçek zamanlı çıkarım motoru
│   └── CMakeLists.txt       # Derleme yapılandırması
├── scripts/
│   ├── collect_data.py      # Kamera üzerinden veri toplama betiği
│   └── train_model.py       # Fine-tuning ve TFLite dönüşüm betiği
├── models/
│   └── model.tflite         # Eğitilmiş Edge AI modeli
└── README.md
