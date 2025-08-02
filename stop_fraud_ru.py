import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# ... (остальной импорт)
# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')

# Предобработка текста (упрощенная версия для мультиязычной модели)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Zа-яЁё\s]', '', text)  # Сохраняем русские и английские буквы
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Загрузка данных
def load_data(file_path='SMSSpamCollection'):
    try:
        data = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
        data['label'] = data['label'].map({'ham': 0, 'spam': 1})
        data['processed_text'] = data['message'].apply(preprocess_text)
        return data
    except FileNotFoundError:
        print("Файл SMSSpamCollection не найден.")
        return None

# Подготовка данных для модели
def prepare_dataset(data):
    if data is None:
        return None
    dataset = Dataset.from_pandas(data[['processed_text', 'label']])
    return dataset

# Токенизация
def tokenize_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    return tokenizer(examples['processed_text'], padding="max_length", truncation=True, max_length=128)

# Модель и обучение
def train_model(dataset):
    if dataset is None:
        return None

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",  # Используем правильный параметр
        logging_dir='./logs',
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
    )

    trainer.train()
    return model, AutoTokenizer.from_pretrained("xlm-roberta-base")

# Предсказание
def predict_message(model, tokenizer, message):
    if model is None or tokenizer is None:
        return "Модель не обучена."
    processed_message = preprocess_text(message)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(processed_message, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Переносим на GPU
    outputs = model(**inputs)
    # Переносим logits на CPU и преобразуем в NumPy
    prediction = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)[0]
    return "Спам" if prediction == 1 else "Не спам"

if __name__ == "__main__":
    # Загрузка и подготовка данных
    data = load_data()
    dataset = prepare_dataset(data)
    if dataset is None:
        exit()

    # Обучение модели
    model, tokenizer = train_model(dataset)

    # Тестовые сообщения
    test_messages = [
        "Поздравляем! Вы выиграли 1000$!",
        "Привет, как дела?",
        "Здравствуй, Альберт. Это администрация вашей школы. Ты видел информацию о смене дат окончания летних каникул?"
    ]
    for msg in test_messages:
        result = predict_message(model, tokenizer, msg)
        print(f"Сообщение: '{msg}' → {result}")