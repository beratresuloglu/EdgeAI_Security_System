#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

// --- LNK HATALARINI BİTİREN SİHİRLİ SATIRLAR ---
#ifdef _DEBUG
#pragma comment(lib, "C:/opencv/build/x64/vc16/lib/opencv_world4120d.lib")
#else
#pragma comment(lib, "C:/opencv/build/x64/vc16/lib/opencv_world4120.lib")
#endif
// Not: Eğer klasöründeki dosya 4100 değilse (örneğin 480 veya 4100d), 
// o klasöre gidip ismini tam kontrol et ve burayı ona göre güncelle.
// ----------------------------------------------
int main() {
    // 1. Modeli Yükle
    // Not: OpenCV 4.x+ sürümleri TFLite dosyalarını readNet ile doğrudan tanır.
    cv::dnn::Net net = cv::dnn::readNet("model.tflite");

    if (net.empty()) {
        std::cerr << "Hata: model.tflite dosyasi bulunamadi!" << std::endl;
        return -1;
    }

    // Backend ayarları (İşlemci üzerinde hızlı çalışması için)
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::VideoCapture cap(0);
    cv::Mat frame;

    std::cout << "Sistem Hazir. Cikmak icin 'q' tusuna basin." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // 2. Görüntü Ön İşleme (Preprocessing)
        // Görüntüyü 224x224 yapar, 1/255 ile normalize eder ve BGR'dan RGB'ye çevirir.
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(0, 0, 0), true, false);
        net.setInput(blob);

        // 3. Yapay Zekayı Çalıştır (Inference)
        cv::Mat detections = net.forward();

        // 4. Sonucu Analiz Et
        float confidence = detections.at<float>(0, 0);

        // Eşik değeri (Python'daki ile aynı: 0.5)
        std::string label;
        cv::Scalar color;

        if (confidence < 0.05) {
            label = "ONAYLANDI: BERAT (" + std::to_string(int((1 - confidence) * 100)) + "%)";
            color = cv::Scalar(0, 255, 0); // Yeşil
        }
        else {
            label = "ERISIM REDDEDILDI";
            color = cv::Scalar(0, 0, 255); // Kırmızı
        }

        // 5. Ekrana Çizdir
        cv::putText(frame, label, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        cv::imshow("Berat_EdgeAI_C++_Final", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}