"""
Fine-tune a transformer model for ISIC classification
Uses DistilBERT for good balance of speed and accuracy
"""
import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pandas as pd
from datetime import datetime

class ISICDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        label = example['label_id']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    """Compute accuracy and F1 score"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

def load_training_data(file_path='isic_training_data.json'):
    """Load prepared training data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def fine_tune_model():
    """Main fine-tuning function"""
    print("Loading training data...")
    training_data = load_training_data()
    
    print(f"Training examples: {len(training_data['train'])}")
    print(f"Validation examples: {len(training_data['validation'])}")
    print(f"Number of classes: {training_data['num_labels']}")
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=training_data['num_labels']
    )
    
    # Create datasets
    train_dataset = ISICDataset(training_data['train'], tokenizer)
    val_dataset = ISICDataset(training_data['validation'], tokenizer)
    
    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./isic_model_{timestamp}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting training...")
    # Train the model
    trainer.train()
    
    # Save the final model
    final_model_dir = f"./isic_classifier_final_{timestamp}"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Save label mappings
    with open(f"{final_model_dir}/label_mappings.json", 'w') as f:
        json.dump({
            'label_to_id': training_data['label_to_id'],
            'id_to_label': training_data['id_to_label']
        }, f, indent=2)
    
    print(f"Model saved to {final_model_dir}")
    
    # Evaluate on validation set
    print("Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Get detailed predictions for analysis
    predictions = trainer.predict(val_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Convert to ISIC codes for analysis
    id_to_label = training_data['id_to_label']
    pred_isic = [id_to_label[str(pred)] for pred in pred_labels]
    true_isic = [id_to_label[str(true)] for true in true_labels]
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(true_isic, pred_isic))
    
    # Save evaluation results
    eval_report = {
        'timestamp': timestamp,
        'model_path': final_model_dir,
        'training_stats': training_data['stats'],
        'eval_results': eval_results,
        'detailed_metrics': classification_report(true_isic, pred_isic, output_dict=True)
    }
    
    with open(f"{final_model_dir}/evaluation_report.json", 'w') as f:
        json.dump(eval_report, f, indent=2, default=str)
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"F1 Score: {eval_results['eval_f1']:.4f}")
    print(f"Model saved to: {final_model_dir}")
    
    return final_model_dir, eval_results

def test_model(model_path, test_texts):
    """Test the fine-tuned model on sample texts"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load label mappings
    with open(f"{model_path}/label_mappings.json", 'r') as f:
        label_mappings = json.load(f)
    
    id_to_label = label_mappings['id_to_label']
    
    print("Testing model predictions:")
    for text in test_texts:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][predicted_class].item()
        
        predicted_isic = id_to_label[str(predicted_class)]
        print(f"Text: {text}")
        print(f"Predicted ISIC: {predicted_isic} (confidence: {confidence:.3f})")
        print()

if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fine-tune the model
    model_path, results = fine_tune_model()
    
    # Test with sample texts
    test_texts = [
        "Manufacturing of steel products",
        "Software development services",
        "Retail clothing store",
        "Restaurant services",
        "Construction of buildings"
    ]
    
    print("\n" + "="*50)
    test_model(model_path, test_texts)