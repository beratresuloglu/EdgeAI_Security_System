from sklearn.datasets import fetch_lfw_people
import cv2
import os

# 1. Klasörü hazırla
save_path = "data/diger"
os.makedirs(save_path, exist_ok=True)

print("İnternetten negatif yüz örnekleri çekiliyor (LFW)...")

# 2. Veri setini indir (Sadece bir kez indirir, yaklaşık 200MB)
lfw_people = fetch_lfw_people(min_faces_per_person=0, resize=1.0)

# 3. İlk 500 yüzü al ve kaydet
count = 0
for i in range(len(lfw_people.images)):
    if count >= 500: break # 500 tane yeterli
    
    # Görüntü normalize gelmiş olabilir, 0-255 arasına çekiyoruz
    face_img = (lfw_people.images[i] * 255).astype('uint8')
    
    # Modeli eğittiğimiz boyuta (224x224) getiriyoruz
    face_resized = cv2.resize(face_img, (224, 224))
    
    # Kaydet
    cv2.imwrite(f"{save_path}/neg_{count}.jpg", face_resized)
    count += 1

print(f"Tamamlandı! {count} adet yeni insan yüzü 'data/diger' klasörüne eklendi.")