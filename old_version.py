import pandas as pd
import numpy as np
import nltk, re, torch
from datasets import load_dataset, Dataset
from nltk.tokenize import word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

nltk.download("punkt")

# 1. Загрузка и предварительная очистка русскоязычного датасета
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

# 3. Обучение модели
def train():
    dataset = load_ru_spam()
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir="./ru_spam_results",
        num_train_epochs=2,
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
    return model

# 4. Предсказание для произвольного текста
def predict(model, text: str) -> str:
    device = next(model.parameters()).device
    proc = preprocess_text(text)
    inputs = tok(proc, return_tensors="pt", truncation=True, padding=True, max_length=96)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Перемещаем входные данные на устройство модели
    with torch.no_grad():
        logits = model(**inputs).logits
    label = int(torch.argmax(logits, dim=1))
    return "⚠️ СПАМ / МОШЕННИЧЕСТВО" if label == 1 else "✅ НОРМАЛЬНОЕ"

# 5. CLI- цикл
if __name__ == "__main__":
    print("⏳ Скачиваем датасет и дообучаем модель…")
    mdl = train()
    print("\n✅ Обучение завершено. Пишите сообщение (выход — пустая строка):")
    while True:
        msg = input("➤ ")
        if not msg.strip():
            break
        print("→", predict(mdl, msg), "\n")