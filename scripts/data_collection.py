import cv2
import os
import time

def collect_data(label, count_limit=2000):
    # Ana dizine göre veri klasörünü ayarla
    save_path = f"data/{label}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Mevcut dosya sayısını kontrol et (Üzerine yazmamak, ekleme yapmak için)
    existing_files = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    start_index = len(existing_files)
    
    # Kamerayı başlat
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Kamera açılmadı!")
        return

    print(f"\n--- {label.upper()} kategorisi için EK VERİ kaydı başlıyor ---")
    print(f"Mevcut kayıt sayısı: {start_index}")
    print("Hazırlanmak için 3 saniyen var...")
    time.sleep(3)
    
    added_count = 0
    while added_count < count_limit:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Görüntüyü kare formatına (224x224) getir
        roi = cv2.resize(frame, (224, 224))
        
        # Ekranda bilgileri göster
        display_frame = roi.copy()
        cv2.putText(display_frame, f"Yeni Kayit: {added_count}/{count_limit}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Toplam: {start_index + added_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Veri Toplama Paneli", display_frame)
        
        # Fotoğrafı kaydet (Dosya ismini start_index'e göre ayarlar)
        current_id = start_index + added_count
        cv2.imwrite(f"{save_path}/{label}_{current_id}.jpg", roi)
        
        added_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Kullanıcı tarafından durduruldu.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print(f"Bitti! {added_count} adet yeni fotoğraf '{save_path}' dizinine eklendi.")
    print(f"Klasördeki toplam '{label}' görseli: {start_index + added_count}")

if __name__ == "__main__":
    # count_limit'i ihtiyacına göre artırabilirsin (Örn: 500)
    collect_data("berat", count_limit=2000) # Berat için 2000 fotoğraf topla
    
    # 'diger' kısmı devre dışı bırakıldı.
    print("\nİşlem tamamlandı. 'diger' klasörüne dokunulmadı.")