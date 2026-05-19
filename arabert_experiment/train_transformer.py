
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

from preprocessing import ArabicTextPreprocessor
from utils import save_object

def main():
    print("=" * 80)
    print("       ARABIC POETRY CLASSIFICATION - ARABERT TRANSFORMER PIPELINE")
    print("=" * 80)

    # --------------------------
    # STEP 1: Configuration (CPU Optimized)
    # --------------------------
    MODEL_NAME = "aubmindlab/bert-base-arabertv2"
    MAX_LEN = 96
    BATCH_SIZE = 8
    EPOCHS = 2
    LEARNING_RATE = 3e-5
    OUTPUT_DIR = "./arabert_model"
    MAX_SAMPLES_PER_CLASS = 1500

    print(f"\n[1/8] Configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max Length: {MAX_LEN}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")

    # --------------------------
    # STEP 2: Load and Preprocess Data
    # --------------------------
    print("\n[2/8] Loading dataset...")
    data = pd.read_csv('Arabic_Poetry_Dataset.csv')
    data = data.dropna().reset_index(drop=True)
    print(f"  Loaded {len(data)} poems")

    print("\n[3/8] Balancing dataset (CPU-friendly)...")
    balanced_data = []
    for era in data['poet_era'].unique():
        era_samples = data[data['poet_era'] == era]
        if len(era_samples) > MAX_SAMPLES_PER_CLASS:
            balanced_data.append(era_samples.sample(MAX_SAMPLES_PER_CLASS, random_state=42))
        else:
            balanced_data.append(era_samples)
    data = pd.concat(balanced_data).reset_index(drop=True)
    print(f"  After balancing: {len(data)} poems")

    print("\n[4/8] Preprocessing text...")
    preprocessor = ArabicTextPreprocessor()
    data['cleaned_text'] = data['poem_text'].apply(preprocessor.clean_text)
    data = data[data['cleaned_text'].str.len() > 10].reset_index(drop=True)
    print(f"  After cleaning: {len(data)} poems")

    # Encode labels
    encoder = LabelEncoder()
    data['label'] = encoder.fit_transform(data['poet_era'])
    num_labels = len(encoder.classes_)
    print(f"  Number of classes: {num_labels}")

    # Split data
    train_df, val_df = train_test_split(
        data[['cleaned_text', 'label']],
        test_size=0.2,
        random_state=42,
        stratify=data['label']
    )
    print(f"  Train size: {len(train_df)}, Val size: {len(val_df)}")

    # --------------------------
    # STEP 3: Load Tokenizer and Model
    # --------------------------
    print("\n[5/8] Loading AraBERT tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    # --------------------------
    # STEP 4: Tokenize Data
    # --------------------------
    print("\n[6/8] Tokenizing data...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["cleaned_text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # --------------------------
    # STEP 5: Training Arguments (Fixed for Compatibility)
    # --------------------------
    print("\n[7/8] Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        weight_decay=0.01,
        warmup_steps=100,
        seed=42,
        data_seed=42,
        report_to="none"
    )

    # --------------------------
    # STEP 6: Train Model
    # --------------------------
    print("\n[8/8] Training AraBERT model...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    trainer.train()

    # --------------------------
    # STEP 7: Evaluate
    # --------------------------
    print("\n" + "=" * 80)
    print("                      EVALUATING BEST MODEL")
    print("=" * 80)
    
    predictions = trainer.predict(tokenized_val)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=encoder.classes_))

    # --------------------------
    # STEP 8: Save Everything
    # --------------------------
    print("\n" + "=" * 80)
    print("                      SAVING MODEL AND OBJECTS")
    print("=" * 80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    save_object(encoder, 'label_encoder_transformer.pkl')
    save_object(OUTPUT_DIR, 'model_path.pkl')
    
    print(f"\n✓ Model saved to: {OUTPUT_DIR}")
    print("✓ Tokenizer saved")
    print("✓ Label encoder saved")
    print("\n" + "=" * 80)
    print("                     TRANSFORMER TRAINING COMPLETE! 🎉")
    print("=" * 80)

if __name__ == "__main__":
    main()

