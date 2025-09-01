"""
Test accuracy of both embedding and fine-tuned models using test dataset
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from fine_tuned_classifier import FineTunedISICClassifier, get_classifier
from sklearn.metrics import accuracy_score, classification_report
import time
import torch

def load_isic_data_for_qdrant():
    """Load ISIC data and create Qdrant collection"""
    print("Loading ISIC data for Qdrant...")
    df = pd.read_excel('Localised ISIC.xlsx', sheet_name='ISIC-Rev4')
    df = df[df['ISIC-Code'].astype(str).str.len() == 4]
    
    # Create text data
    df["text"] = df["ISIC-Sub Activity Description"].fillna("")
    data_dict = df[["ISIC-Code", "ISIC-Sub Activity Description", "text"]].rename(
        columns={"ISIC-Sub Activity Description": "title", "ISIC-Code": "code"}
    ).to_dict(orient="records")
    
    return df, data_dict

def setup_qdrant_collection(encoder, data_dict):
    """Setup Qdrant collection with ISIC data"""
    print("Setting up Qdrant collection...")
    qdrant = QdrantClient(":memory:")
    
    # Create collection
    qdrant.recreate_collection(
        collection_name="industries",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    
    # Encode and upsert data
    texts = [record['text'] for record in data_dict]
    print(f"Encoding {len(texts)} ISIC descriptions...")
    vectors = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    
    # Upsert to Qdrant
    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "code": data_dict[i]['code'],
                "title": data_dict[i]['title']
            }
        ) for i in range(len(vectors))
    ]
    
    qdrant.upsert(collection_name="industries", points=points)
    print(f"Uploaded {len(points)} vectors to Qdrant")
    
    return qdrant

def load_test_data(file_path):
    """Load and validate test data"""
    print(f"Loading test data from {file_path}...")
    
    try:
        if file_path.endswith('.xlsx'):
            test_df = pd.read_excel(file_path)
        else:
            test_df = pd.read_csv(file_path)
        
        print(f"Loaded {len(test_df)} test records")
        print(f"Columns: {', '.join(test_df.columns.tolist())}")
        
        # Find expected ISIC column
        expected_col = None
        possible_expected_cols = ['ISIC', 'ISIC_CODE', 'EXPECTED_ISIC', 'TRUE_ISIC', 'ACTUAL_ISIC']
        for col in possible_expected_cols:
            if col in test_df.columns:
                expected_col = col
                break
        
        if not expected_col:
            raise ValueError(f"No expected ISIC column found. Looking for one of: {', '.join(possible_expected_cols)}")
        
        if 'INDUSTRY' not in test_df.columns:
            raise ValueError("Missing 'INDUSTRY' column")
        
        # Clean and filter data
        original_len = len(test_df)
        test_df = test_df.dropna(subset=['INDUSTRY', expected_col])
        test_df = test_df[test_df['INDUSTRY'].str.len() > 2]
        test_df = test_df[test_df[expected_col].astype(str).str.len() == 4]
        
        print(f"Filtered to {len(test_df)} valid records (removed {original_len - len(test_df)} invalid)")
        
        return test_df, expected_col
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None

def test_embedding_model(test_industries, true_codes, encoder, qdrant):
    """Test embedding model accuracy"""
    print("\nTesting embedding model...")
    start_time = time.time()
    
    predictions = []
    for i, industry in enumerate(test_industries):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(test_industries)}")
        
        vector = encoder.encode(industry).tolist()
        hits = qdrant.search(collection_name="industries", query_vector=vector, limit=1)
        pred_code = hits[0].payload.get('code') if hits else None
        predictions.append(pred_code)
    
    accuracy = accuracy_score(true_codes, predictions)
    elapsed = time.time() - start_time
    
    print(f"Embedding model - Accuracy: {accuracy:.3f} ({accuracy:.1%}) in {elapsed:.1f}s")
    return predictions, accuracy

def test_fine_tuned_model(test_industries, true_codes, fine_tuned_classifier):
    """Test fine-tuned model accuracy"""
    print("\nTesting fine-tuned model...")
    start_time = time.time()
    
    predictions = []
    for i, industry in enumerate(test_industries):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(test_industries)}")
        
        pred_results = fine_tuned_classifier.predict_single(industry, top_k=1)
        pred_code = pred_results[0]['code'] if pred_results else None
        predictions.append(pred_code)
    
    accuracy = accuracy_score(true_codes, predictions)
    elapsed = time.time() - start_time
    
    print(f"Fine-tuned model - Accuracy: {accuracy:.3f} ({accuracy:.1%}) in {elapsed:.1f}s")
    return predictions, accuracy

def analyze_results(test_df, expected_col, embedding_preds, finetuned_preds, embedding_acc, finetuned_acc):
    """Analyze and save detailed results"""
    print("\n" + "="*60)
    print("ACCURACY COMPARISON RESULTS")
    print("="*60)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'Industry': test_df['INDUSTRY'].tolist(),
        'True_ISIC': test_df[expected_col].astype(str).tolist(),
        'Embedding_Prediction': embedding_preds,
        'FineTuned_Prediction': finetuned_preds,
        'Embedding_Correct': [str(t) == str(p) for t, p in zip(test_df[expected_col].astype(str), embedding_preds)],
        'FineTuned_Correct': [str(t) == str(p) for t, p in zip(test_df[expected_col].astype(str), finetuned_preds)]
    })
    
    # Summary metrics
    print(f"Test Records: {len(results_df)}")
    print(f"Embedding Model Accuracy: {embedding_acc:.3f} ({embedding_acc:.1%})")
    print(f"Fine-tuned Model Accuracy: {finetuned_acc:.3f} ({finetuned_acc:.1%})")
    print(f"Improvement: {finetuned_acc - embedding_acc:+.3f} ({(finetuned_acc - embedding_acc):.1%} points)")
    
    # Cases where fine-tuned is better
    better_cases = results_df[
        (~results_df['Embedding_Correct']) & 
        (results_df['FineTuned_Correct'])
    ]
    
    # Cases where embedding is better
    worse_cases = results_df[
        (results_df['Embedding_Correct']) & 
        (~results_df['FineTuned_Correct'])
    ]
    
    print(f"\nDetailed Analysis:")
    print(f"  Cases where fine-tuned is better: {len(better_cases)}")
    print(f"  Cases where embedding is better: {len(worse_cases)}")
    print(f"  Both correct: {len(results_df[(results_df['Embedding_Correct']) & (results_df['FineTuned_Correct'])])}")
    print(f"  Both incorrect: {len(results_df[(~results_df['Embedding_Correct']) & (~results_df['FineTuned_Correct'])])}")
    
    # Sample improvements
    if len(better_cases) > 0:
        print(f"\nSample cases where fine-tuned model is better:")
        for _, row in better_cases.head(5).iterrows():
            print(f"  Industry: {row['Industry'][:50]}...")
            print(f"    True: {row['True_ISIC']}, Embedding: {row['Embedding_Prediction']}, Fine-tuned: {row['FineTuned_Prediction']}")
    
    # Save detailed results
    timestamp = int(time.time())
    output_file = f"accuracy_comparison_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")
    
    return results_df

def main():
    """Main function to run accuracy tests"""
    print("ISIC Model Accuracy Testing")
    print("="*40)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize models
    print("\n1. Loading models...")
    encoder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    
    fine_tuned_classifier = get_classifier()
    if not fine_tuned_classifier:
        print("Fine-tuned classifier not found!")
        return
    
    print("Models loaded successfully")
    
    # Load ISIC data and setup Qdrant
    print("\n2. Setting up ISIC database...")
    isic_df, data_dict = load_isic_data_for_qdrant()
    qdrant = setup_qdrant_collection(encoder, data_dict)
    print("ISIC database ready")
    
    # Load test data
    print("\n3. Loading test data...")
    test_file = "d:/LMS/isic_test_v2.xlsx"
    test_df, expected_col = load_test_data(test_file)
    
    if test_df is None:
        print("Failed to load test data")
        return
    
    test_industries = test_df['INDUSTRY'].tolist()
    true_codes = test_df[expected_col].astype(str).tolist()
    print("Test data ready")
    
    # Run tests
    print("\n4. Running accuracy tests...")
    
    # Test embedding model
    embedding_preds, embedding_acc = test_embedding_model(test_industries, true_codes, encoder, qdrant)
    
    # Test fine-tuned model
    finetuned_preds, finetuned_acc = test_fine_tuned_model(test_industries, true_codes, fine_tuned_classifier)
    
    # Analyze results
    print("\n5. Analyzing results...")
    results_df = analyze_results(test_df, expected_col, embedding_preds, finetuned_preds, embedding_acc, finetuned_acc)
    
    print("\nAccuracy testing complete!")

if __name__ == "__main__":
    main()