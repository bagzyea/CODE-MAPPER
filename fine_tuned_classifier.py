"""
Fine-tuned ISIC classifier integration module
This module provides a wrapper for the fine-tuned transformer model
"""
import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob
from typing import List, Tuple, Dict

class FineTunedISICClassifier:
    def __init__(self, model_path: str = None):
        self.model = None
        self.tokenizer = None
        self.label_mappings = None
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path:
            self.load_model(model_path)
    
    def find_latest_model(self) -> str:
        """Find the most recent fine-tuned model directory"""
        model_dirs = glob.glob("./isic_classifier_final_*")
        if not model_dirs:
            raise FileNotFoundError("No fine-tuned model found. Please run fine_tune_model.py first.")
        
        # Sort by timestamp in filename and return latest
        latest_model = sorted(model_dirs)[-1]
        return latest_model
    
    def load_model(self, model_path: str = None):
        """Load the fine-tuned model and tokenizer"""
        if model_path is None:
            model_path = self.find_latest_model()
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        
        print(f"Loading fine-tuned model from {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        mappings_path = os.path.join(model_path, "label_mappings.json")
        if os.path.exists(mappings_path):
            with open(mappings_path, 'r') as f:
                self.label_mappings = json.load(f)
        else:
            raise FileNotFoundError(f"Label mappings not found at {mappings_path}")
        
        self.model_path = model_path
        print(f"Model loaded successfully. Device: {self.device}")
    
    def predict_single(self, text: str, top_k: int = 5) -> List[Dict]:
        """
        Predict ISIC code for a single text
        Returns list of top_k predictions with confidence scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top k predictions
        top_predictions = torch.topk(predictions[0], k=min(top_k, predictions.size(1)))
        
        results = []
        for i in range(len(top_predictions.values)):
            confidence = top_predictions.values[i].item()
            predicted_class_id = top_predictions.indices[i].item()
            isic_code = self.label_mappings['id_to_label'][str(predicted_class_id)]
            
            results.append({
                'code': isic_code,
                'confidence': confidence,
                'title': f"ISIC {isic_code}"  # You might want to load descriptions
            })
        
        return results
    
    def predict_batch(self, texts: List[str], top_k: int = 1) -> List[List[Dict]]:
        """
        Predict ISIC codes for a batch of texts
        Returns list of predictions for each text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        results = []
        # Process in batches for memory efficiency
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                prediction = self.predict_single(text, top_k=top_k)
                batch_results.append(prediction)
            
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        # Try to load evaluation report if available
        eval_report_path = os.path.join(self.model_path, "evaluation_report.json")
        eval_info = {}
        
        if os.path.exists(eval_report_path):
            with open(eval_report_path, 'r') as f:
                eval_report = json.load(f)
                eval_info = {
                    'accuracy': eval_report.get('eval_results', {}).get('eval_accuracy', 'N/A'),
                    'f1_score': eval_report.get('eval_results', {}).get('eval_f1', 'N/A'),
                    'training_examples': eval_report.get('training_stats', {}).get('total_examples', 'N/A')
                }
        
        return {
            'status': 'Loaded',
            'model_path': self.model_path,
            'device': str(self.device),
            'num_labels': len(self.label_mappings['id_to_label']),
            'model_type': 'Fine-tuned DistilBERT',
            'evaluation': eval_info
        }

# Global classifier instance
_classifier = None

def get_classifier() -> FineTunedISICClassifier:
    """Get or create global classifier instance"""
    global _classifier
    if _classifier is None:
        _classifier = FineTunedISICClassifier()
        try:
            _classifier.load_model()
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            return None
    return _classifier

def classify_text(text: str, top_k: int = 5) -> List[Dict]:
    """Convenience function to classify text using global classifier"""
    classifier = get_classifier()
    if classifier is None:
        return []
    return classifier.predict_single(text, top_k=top_k)

# Test function
if __name__ == "__main__":
    # Test the classifier
    classifier = FineTunedISICClassifier()
    
    try:
        classifier.load_model()
        
        # Test predictions
        test_texts = [
            "Manufacturing of steel products",
            "Software development services", 
            "Retail clothing store",
            "Restaurant services",
            "Construction of buildings"
        ]
        
        print("Testing Fine-tuned ISIC Classifier:")
        print("=" * 50)
        
        for text in test_texts:
            predictions = classifier.predict_single(text, top_k=3)
            print(f"\nText: {text}")
            print("Predictions:")
            for i, pred in enumerate(predictions, 1):
                print(f"  {i}. ISIC {pred['code']} (confidence: {pred['confidence']:.3f})")
        
        # Print model info
        print("\nModel Information:")
        info = classifier.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the fine-tuning script first: python fine_tune_model.py")