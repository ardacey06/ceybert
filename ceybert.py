import gradio as gr
from transformers import pipeline

# 1. Modeli Yükle
model_path = "./final_sentiment_model"
sentiment_analysis = pipeline("text-classification", model=model_path, tokenizer=model_path, top_k=None)

# 2. Duygu Çevirileri 
emoji_map = {
    "mutlu": "😊 Mutlu",
    "üzgün": "😔 Üzgün",
    "kızgın": "😡 Kızgın",
    "sürpriz": "😮 Sürpriz",
    "tiksinti": "🤢 Tiksinti",
    "korku": "😱 Korku",
    "label_0": "😶 Bilinmiyor" # Nötr veya Belirsiz
}

def analyze_sentiment(text):
    # Model tahmini
    results = sentiment_analysis(text)[0] 
    
    #Gradio {Label: Score} formatı
    output_dict = {}
    for result in results:
        label = result['label']
        score = result['score']
        
        # İkon ekleme
        display_label = emoji_map.get(label, label)
        output_dict[display_label] = score
    
    # Eşik Değeri Kontrolü (Threshold Logic)
    top_score = max(output_dict.values())
    
    # Eğer en yüksek skor %60'ın altındaysa (Model emin değilse)
    if top_score < 0.60:
        return {"😶 Nötr / Belirsiz": 1.0}
        
    return output_dict

# 3. Arayüzü Oluştur
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🇹🇷 TurkishBERTiment: Türkçe Duygu Analizi
    Bu model, Türkçe metinlerdeki duygusal durumu (Mutlu, Üzgün, Kızgın, Sürpriz, Tiksinti, Korku) analiz etmek için **BERTurk** kullanılarak eğitilmiştir.
    """)
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Analiz edilecek tümceyi yazın", 
                placeholder="Örn: Bu ürün harika ama kargo biraz gecikti...",
                lines=3
            )
            analyze_btn = gr.Button("Analiz Et", variant="primary")
            
        with gr.Column():
            label_output = gr.Label(label="Duygu Durumu", num_top_classes=3)
    
    # Örnek butonlar
    examples = [
        ["Sınavdan yüz aldığımı görünce havalara uçtum!"],
        ["Bu yemeğin tadı gerçekten berbat."],
        ["Gördüklerinden sonra küplere bindi."]
    ]
    gr.Examples(examples=examples, inputs=input_text)

    analyze_btn.click(fn=analyze_sentiment, inputs=input_text, outputs=label_output)

# 4. Uygulamayı Başlat
if __name__ == "__main__":
    demo.launch()