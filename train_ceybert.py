import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# --- 1. Veriyi Hazırlama ---
print("Veri yükleniyor...")
df = pd.read_csv('final_train_data.csv')

# Etiketleri sayısal hale getirme
unique_labels = df['label'].unique()
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

df['label'] = df['label'].map(label2id)

# Train ve Test (%90 Train, %10 Test)
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

# Hugging Face Dataset formatına çevirme
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)

# --- 2. Tokenizer ve Model ---
model_name = "dbmdz/bert-base-turkish-cased"
print(f"{model_name} indiriliyor...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id
)

# Data Tokenization
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

print("Veriler tokenize ediliyor...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# --- 3. Metrikler (Başarı Ölçümü) ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 4. Eğitim Ayarları ---
training_args = TrainingArguments(
    output_dir="./sonuc_model",
    num_train_epochs=3,              # Veri üzerinden kaç kere geçecek (3 ideal)
    per_device_train_batch_size=8,   # GPU belleği için düşük tutuldu
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,   # Küçük batch'i telafi etmek için biriktirme
    eval_strategy="epoch",           # Her epoch sonunda test et
    save_strategy="epoch",
    learning_rate=2e-5,
    fp16=True,                       # CUDA hızlandırma ve bellek tasarrufu 
    load_best_model_at_end=True,
    logging_steps=50,                # Her 50 adımda terminale bilgi bas
    report_to="none"                 # Wandb vs. kapalı
)

# --- 5. Eğitimi Başlat ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

print("Eğitim başlıyor! Bu işlem süresi GPU hızına göre değişebilir...")
trainer.train(resume_from_checkpoint=True)

# Modeli Kaydet
print("Model kaydediliyor...")
model.save_pretrained("./final_sentiment_model")
tokenizer.save_pretrained("./final_sentiment_model")
print("İşlem tamam! Model './final_sentiment_model' klasörüne kaydedildi.")