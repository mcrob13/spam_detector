import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Загрузка модели и токенизатора
def load_model(model_path="C:\\Users\\robog\\PycharmProjects\\tester\\ru_spam_model"):
    print(f"Проверяем путь: {os.path.abspath(model_path)}")
    if os.path.exists(model_path) and (os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or os.path.exists(os.path.join(model_path, "model.safetensors"))):
        print(f"Загружаем модель из {model_path}…")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        raise FileNotFoundError(f"Модель не найдена в {model_path}. Запустите train_model.py для обучения.")
    return model, tokenizer

# 2. Пред-обработка текста
def preprocess_text(txt):
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

# 3. Предсказание
def predict(model, tokenizer, text: str) -> str:
    device = next(model.parameters()).device
    proc = preprocess_text(text)
    inputs = tokenizer(proc, return_tensors="pt", truncation=True, padding=True, max_length=96)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    label = int(torch.argmax(logits, dim=1))
    return "⚠️ СПАМ / МОШЕННИЧЕСТВО" if label == 1 else "✅ НОРМАЛЬНОЕ"
# 4. CLI-цикл
if __name__ == "__main__":
    try:
        mdl, tkn = load_model()
        print("\n✅ Модель загружена. Пишите сообщение (выход — пустая строка):")
        while True:
            msg = input("➤ ")
            if not msg.strip():
                break
            print("→", predict(mdl, tkn, msg), "\n")
    except FileNotFoundError as e:
        print(e)