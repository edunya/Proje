!pip install datasets

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

os.environ["WANDB_DISABLED"] = "true"

file_path = "/content/Combined_Articles.xlsx"
data = pd.read_excel(file_path)

# Metin Alanlarını Birleştirme
texts = (data['Baslik'].fillna('') + ' ' +
         data['Ozet'].fillna('') + ' ' +
         data['Anahtar_Kelimeler'].fillna('')).astype(str).tolist()
labels = data['Topic'].tolist()

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Eğitim ve Test Verilerini Ayırma
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Paralelleştirme
train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)

# Model ve Eğitim Argümanları
if os.path.exists("./mb_model"):
    model = BertForSequenceClassification.from_pretrained("./mb_model")
    tokenizer = BertTokenizer.from_pretrained("./mb_model")
else:
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", num_labels=len(label_encoder.classes_)
    )

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    learning_rate=3e-6,
    weight_decay=0.01,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    gradient_accumulation_steps=2,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
)

if not os.path.exists("./mb_model"):
    start_train_time = time.time()
    trainer.train()
    end_train_time = time.time()
    model.save_pretrained("./mb_model")
    tokenizer.save_pretrained("./mb_model")
    training_time = end_train_time - start_train_time
    print(f"Training Time: {training_time:.2f} seconds")
else:
    print("Model loaded. Skipping training.")

# Tahmin ve Değerlendirme
start_inference_time = time.time()
predictions = trainer.predict(test_dataset)
end_inference_time = time.time()

inference_time = end_inference_time - start_inference_time
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

cm = confusion_matrix(y_true, y_pred)
sensitivity = cm.diagonal() / cm.sum(axis=1)
specificity = np.sum(cm) - cm.sum(axis=1) - cm.sum(axis=0) + cm.diagonal()
specificity = specificity / (np.sum(cm) - cm.sum(axis=1))

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall (Sensitivity): {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Specificity (class-wise): {specificity}")
print(f"Inference Time: {inference_time:.2f} seconds")

plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(np.arange(len(label_encoder.classes_)), label_encoder.classes_)
plt.show()

probs = predictions.predictions
fpr, tpr, roc_auc = {}, {}, {}

for i in range(len(label_encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(len(label_encoder.classes_)):
    plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Eğitim ve Test Loss Grafiği
metrics = trainer.state.log_history
train_loss = [m["loss"] for m in metrics if "loss" in m]
eval_loss = [m["eval_loss"] for m in metrics if "eval_loss" in m]

plt.plot(train_loss, label="Train Loss")
plt.plot(eval_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.show()
