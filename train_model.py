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

# 3. Обучение модели
def train_model(model_path="C:\\Users\\robog\\PycharmProjects\\tester\\ru_spam_model"):
    print(f"Обучаем модель и сохраняем в {model_path}…")
    dataset = load_ru_spam()
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    args = TrainingArguments(
        output_dir="./ru_spam_results",
        num_train_epochs=2,  # Уменьшено для ускорения
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
    print("Обучение завершено. Сохраняем модель…")
    model.save_pretrained(model_path)
    tok.save_pretrained(model_path)
    print(f"Модель и токенизатор сохранены в {model_path}")

if __name__ == "__main__":
    train_model()