"""
Quick comparison test between embedding and fine-tuned models
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from fine_tuned_classifier import FineTunedISICClassifier, get_classifier
import time
import torch

def setup_models():
    """Initialize both models"""
    print("Loading models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    
    fine_tuned_classifier = get_classifier()
    if not fine_tuned_classifier:
        print("Fine-tuned classifier not found!")
        return None, None, None
    
    # Setup Qdrant with ISIC data
    print("Setting up ISIC database...")
    df = pd.read_excel('Localised ISIC.xlsx', sheet_name='ISIC-Rev4')
    df = df[df['ISIC-Code'].astype(str).str.len().isin([2, 4])]
    df["text"] = df["ISIC-Sub Activity Description"].fillna("")
    
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="industries",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    
    # Encode and upload data
    texts = df["text"].tolist()
    codes = df["ISIC-Code"].astype(str).tolist()
    titles = df["ISIC-Sub Activity Description"].fillna("").tolist()
    
    print(f"Encoding {len(texts)} ISIC descriptions...")
    vectors = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    points = [
        models.PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={"code": codes[i], "title": titles[i]}
        ) for i in range(len(vectors))
    ]
    
    qdrant.upsert(collection_name="industries", points=points)
    print("ISIC database ready")
    
    return encoder, qdrant, fine_tuned_classifier

def compare_models(test_texts, encoder, qdrant, fine_tuned_classifier):
    """Compare both models on test texts"""
    print(f"\nComparing models on {len(test_texts)} test cases...")
    
    results = []
    
    for i, text in enumerate(test_texts):
        if i % 20 == 0:
            print(f"  Processing {i+1}/{len(test_texts)}")
        
        # Embedding model prediction
        start_time = time.time()
        vector = encoder.encode(text).tolist()
        hits = qdrant.search(collection_name="industries", query_vector=vector, limit=3)
        embedding_codes = [hit.payload.get('code') for hit in hits]
        embedding_time = time.time() - start_time
        
        # Fine-tuned model prediction
        start_time = time.time()
        ft_predictions = fine_tuned_classifier.predict_single(text, top_k=3)
        ft_codes = [pred['code'] for pred in ft_predictions]
        ft_confidences = [pred['confidence'] for pred in ft_predictions]
        ft_time = time.time() - start_time
        
        results.append({
            'text': text,
            'embedding_top1': embedding_codes[0] if embedding_codes else None,
            'embedding_top3': embedding_codes,
            'finetuned_top1': ft_codes[0] if ft_codes else None,
            'finetuned_top3': ft_codes,
            'finetuned_confidences': ft_confidences,
            'embedding_time_ms': embedding_time * 1000,
            'finetuned_time_ms': ft_time * 1000,
            'agreement_top1': (embedding_codes[0] if embedding_codes else None) == (ft_codes[0] if ft_codes else None)
        })
    
    return results

def analyze_comparison(results):
    """Analyze comparison results"""
    print("\n" + "="*60)
    print("MODEL COMPARISON ANALYSIS")
    print("="*60)
    
    total_cases = len(results)
    agreement_top1 = sum(1 for r in results if r['agreement_top1'])
    
    # Performance metrics
    embedding_avg_time = np.mean([r['embedding_time_ms'] for r in results])
    finetuned_avg_time = np.mean([r['finetuned_time_ms'] for r in results])
    
    # Confidence analysis
    avg_confidence = np.mean([r['finetuned_confidences'][0] if r['finetuned_confidences'] else 0 for r in results])
    
    print(f"Test Cases: {total_cases}")
    print(f"Top-1 Agreement: {agreement_top1}/{total_cases} ({agreement_top1/total_cases:.1%})")
    print(f"Top-1 Disagreement: {total_cases-agreement_top1}/{total_cases} ({1-agreement_top1/total_cases:.1%})")
    
    print(f"\nPerformance:")
    print(f"  Embedding Model: {embedding_avg_time:.1f}ms avg")
    print(f"  Fine-tuned Model: {finetuned_avg_time:.1f}ms avg")
    print(f"  Fine-tuned Average Confidence: {avg_confidence:.3f}")
    
    # Show disagreement cases
    disagreements = [r for r in results if not r['agreement_top1']]
    print(f"\nSample Disagreement Cases ({min(5, len(disagreements))} shown):")
    
    for i, case in enumerate(disagreements[:5]):
        print(f"\n  {i+1}. Text: {case['text'][:60]}...")
        print(f"     Embedding: {case['embedding_top1']}")
        print(f"     Fine-tuned: {case['finetuned_top1']} (confidence: {case['finetuned_confidences'][0]:.3f})")
    
    # High confidence fine-tuned predictions
    high_conf_cases = [r for r in results if r['finetuned_confidences'] and r['finetuned_confidences'][0] > 0.8]
    print(f"\nHigh Confidence Fine-tuned Predictions ({len(high_conf_cases)} cases with >80% confidence):")
    
    for i, case in enumerate(high_conf_cases[:3]):
        print(f"\n  {i+1}. Text: {case['text'][:60]}...")
        print(f"     Prediction: {case['finetuned_top1']} (confidence: {case['finetuned_confidences'][0]:.3f})")
        print(f"     Agreement with embedding: {case['agreement_top1']}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = f"model_comparison_{int(time.time())}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

def main():
    """Main comparison function"""
    print("ISIC Model Comparison Test")
    print("="*30)
    
    # Initialize models
    encoder, qdrant, fine_tuned_classifier = setup_models()
    if not all([encoder, qdrant, fine_tuned_classifier]):
        return
    
    # Load test texts
    test_df = pd.read_excel('d:/LMS/isic_test_v2.xlsx')
    test_texts = test_df['INDUSTRY'].dropna().str.strip()
    test_texts = test_texts[test_texts.str.len() > 2].tolist()
    
    # Take a sample for quick testing
    sample_size = min(100, len(test_texts))
    test_sample = test_texts[:sample_size]
    
    print(f"Using {sample_size} test cases from {len(test_texts)} total")
    
    # Compare models
    results = compare_models(test_sample, encoder, qdrant, fine_tuned_classifier)
    
    # Analyze results
    analyze_comparison(results)
    
    print(f"\nComparison complete! Both models are integrated and working.")

if __name__ == "__main__":
    main()