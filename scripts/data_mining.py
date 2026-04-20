import os
import shutil
import random

# --- AYARLAR ---
lfw_ana_klasor = 'C:\\Users\\BERAT\\Desktop\\WorkPlace\\Berat_EdgeAI\\natural_images\\'
hedef_klasor = 'C:\\Users\\BERAT\\Desktop\\WorkPlace\\Berat_EdgeAI\\data\\diger' # Kendi projenin negatif klasörü
alinacak_maks_resim = 2000 # Kaç adet negatif resim istiyorsun?

# Hedef klasör yoksa oluştur
if not os.path.exists(hedef_klasor):
    os.makedirs(hedef_klasor)

# Tüm resim yollarını topla
tum_resim_yollari = []
for kok_dizin, alt_dizinler, dosyalar in os.walk(lfw_ana_klasor):
    for dosya in dosyalar:
        if dosya.endswith('.jpg') or dosya.endswith('.png'):
            tam_yol = os.path.join(kok_dizin, dosya)
            tum_resim_yollari.append(tam_yol)

# Resimleri karıştır (Farklı kişilerden rastgele gelsin)
random.shuffle(tum_resim_yollari)

# Belirlediğin sayı kadarını kopyala
sayac = 0
for i in range(min(alinacak_maks_resim, len(tum_resim_yollari))):
    kaynak = tum_resim_yollari[i]
    # Dosya adının çakışmaması için isme bir sayı ekleyelim
    dosya_adi = f"negatif_{1999+i}.jpg"
    hedef = os.path.join(hedef_klasor, dosya_adi)
    
    shutil.copy2(kaynak, hedef)
    sayac += 1

print(f"İşlem tamam! {sayac} adet rastgele yüz '{hedef_klasor}' klasörüne kopyalandı.")