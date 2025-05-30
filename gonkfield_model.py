import spacy
from sklearn.preprocessing import LabelEncoder
from spacy.training.example import Example
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
#Trains the model for the voice assistant


#loading JSON file

with open("training_data.json") as f:
    training_data = json.load(f)
    ner_data = training_data["ner_train_data"]  # ✅ loaded from training_data
    intent_data = training_data["intent_train_data"]  # ✅ loaded from training_data

# --- ner model ---
# Create blank model
nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")




# Convert to spaCy format: list of (text, {"entities": [...]})
ner_train_data = [(item["text"], {"entities": [tuple(ent) for ent in item["entities"]]}) for item in ner_data]
# Add entity labels to NER, loops through data and gets the entity labels
for _, annotations in ner_train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])


# Train model
optimizer = nlp.begin_training()
for i in range(20):  # More iterations = better accuracy (to a point)
    for text, annotations in ner_train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], sgd=optimizer)

# Save model
nlp.to_disk("gonkfield_ner_model")
print("NER Model saved to 'gonkfield_ner_model/'")


# --- Intent model training ---


texts = [item["text"] for item in intent_data]
labels = [item["intent"] for item in intent_data]

# Encode string labels into integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Create Hugging Face dataset
dataset = Dataset.from_dict({"text": texts, "label": encoded_labels})

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenize dataset
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./intent_model",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train!
trainer.train()

# Save model, tokenizer, and label encoder
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")

import pickle
with open("./intent_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Intent Model saved to 'intent_model/'")