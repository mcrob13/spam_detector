import os
import pandas as pd
import numpy as np
import nltk
import re
import torch
from datasets import load_dataset, Dataset
from nltk.tokenize import word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

nltk.download("punkt")

# 1. Загрузка и предварительная очистка датасета
def load_ru_spam():
    ds = load_dataset("DmitryKRX/anti_spam_ru", split="train")
    df = pd.DataFrame(ds)
    df.rename(columns={"text": "message", "is_spam": "label"}, inplace=True)
    df["processed"] = df["message"].apply(preprocess_text)
    return Dataset.from_pandas(df[["processed", "label"]])

# 2. Токенизация и пред-обработка текста
tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

def preprocess_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def tokenize(batch):
    return tok(batch["processed"],
               padding="max_length",
               truncation=True,
               max_length=96)

# 3. Обучение или загрузка модели
def train_or_load(model_path="./ru_spam_model"):
    print(f"Проверяем путь: {os.path.abspath(model_path)}")
    if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print(f"Найдена модель в {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        print(f"Модель не найдена в {model_path}. Обучаем новую…")
        dataset = load_ru_spam()
        dataset = dataset.map(tokenize, batched=True)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        args = TrainingArguments(
            output_dir="./ru_spam_results",
            num_train_epochs=1,  # Уменьшено до 1 для ускорения
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=25,
            save_strategy="no",
            learning_rate=2e-5,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tok,
        )
        trainer.train()
        print("Обучение завершено. Сохраняем модель…")  # Отладочный вывод
        # Сохранение модели
        model.save_pretrained("./ru_spam_model")
        tok.save_pretrained("./ru_spam_model")
        tokenizer = tok

    return model, tokenizer

# 4. Предсказание для произвольного текста
def predict(model, tokenizer, text: str) -> str:
    device = next(model.parameters()).device
    proc = preprocess_text(text)
    inputs = tokenizer(proc, return_tensors="pt", truncation=True, padding=True, max_length=96)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    label = int(torch.argmax(logits, dim=1))
    return "⚠️ СПАМ / МОШЕННИЧЕСТВО" if label == 1 else "✅ НОРМАЛЬНОЕ"

# 5. CLI-цикл
if __name__ == "__main__":
    print("⏳ Проверяем наличие модели и запускаем процесс…")
    mdl, tkn = train_or_load()
    print("\n✅ Готово. Пишите сообщение (выход — пустая строка):")
    while True:
        msg = input("➤ ")Ф
        if not msg.strip():
            break
        print("→", predict(mdl, tkn, msg), "\n")