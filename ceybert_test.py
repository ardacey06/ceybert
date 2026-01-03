from transformers import pipeline

# 1. Modeli yükle
model_path = "./final_sentiment_model"
print("Model ve konfigürasyon yükleniyor...")

classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)

# 2. Beş farklı duygu için test cümleleri
test_senaryolari = [
    "Sınavdan yüz aldığımı görünce havalara uçtum!",       # Beklenti: Mutlu
    "Köpeğimin hastalanması beni çok derinden sarstı.",    # Beklenti: Üzgün
    "Benimle böyle konuşmaya nasıl cüret edersin!",        # Beklenti: Kızgın
    "İnanmıyorum! Yıllar sonra seni burada görmek şok edici.", # Beklenti: Sürpriz
    "Bu yemeğin tadı berbat, midem bulandı resmen.",      # Beklenti: Tiksinti
    "Hadi işine bak kardeşim akşam akşam beni yorma.",    # Beklenti: Kızgın
    "Siktir git lan burdan!",                             # Beklenti: Kızgın
    "Bu filmi izlerken çok korktum, kalbim yerinden çıkacak gibiydi.",  # Beklenti: Korku
    "Seni görünce çok sevindim, yıllardır görüşmemiştik!",    # Beklenti: Mutlu
]

print("\n--- Analiz Sonuçları ---\n")

# 3. Tahminleri yap
results = classifier(test_senaryolari)

for metin, sonuc in zip(test_senaryolari, results):
    label = sonuc['label']
    score = sonuc['score']
    
    print(f"Metin: {metin}")
    print(f"Duygu: {label} (Güven: %{score*100:.2f})")
    print("-" * 30)