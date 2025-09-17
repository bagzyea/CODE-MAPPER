"""
Training data preparation for ISIC classification fine-tuning
Creates augmented training examples from the ISIC data
"""
import pandas as pd
import numpy as np
import json
import random
from typing import List, Dict, Tuple

def load_isic_data():
    """Load and preprocess ISIC data"""
    df = pd.read_excel('Localised ISIC.xlsx', sheet_name='ISIC-Rev4')
    # Include both Level 2 (2-digit) and Level 4 (4-digit) codes
    df = df[df['ISIC-Code'].astype(str).str.len().isin([2, 4])]
    return df

def create_text_variations(text: str) -> List[str]:
    """Generate variations of industry descriptions for data augmentation"""
    variations = [text.strip()]
    
    # Basic transformations
    if text.strip():
        # Remove leading spaces and clean
        clean_text = text.strip()
        variations.append(clean_text)
        
        # Add "This involves" prefix
        variations.append(f"This involves {clean_text.lower()}")
        
        # Add "Activities include" prefix  
        variations.append(f"Activities include {clean_text.lower()}")
        
        # Add "Business of" prefix
        variations.append(f"Business of {clean_text.lower()}")
        
        # Add "Company engaged in" prefix
        variations.append(f"Company engaged in {clean_text.lower()}")
    
    return list(set(variations))  # Remove duplicates

def create_training_examples(df: pd.DataFrame) -> List[Dict]:
    """Create training examples with augmentation"""
    examples = []
    
    for _, row in df.iterrows():
        code = str(row['ISIC-Code'])
        description = row['ISIC-Sub Activity Description']
        
        # Skip if description is too short or invalid
        if pd.isna(description) or len(str(description).strip()) < 5:
            continue
            
        # Create variations of the description
        variations = create_text_variations(str(description))
        
        for variation in variations:
            if variation.strip():
                examples.append({
                    'text': variation.strip(),
                    'label': code,
                    'original_description': str(description).strip()
                })
    
    return examples

def create_synthetic_examples(df: pd.DataFrame, num_synthetic: int = 1000) -> List[Dict]:
    """Create synthetic training examples by mixing and matching components"""
    synthetic_examples = []
    
    # Get unique components
    activities = []
    for _, row in df.iterrows():
        desc = str(row['ISIC-Sub Activity Description']).strip()
        if desc and len(desc) > 5:
            activities.append((desc, str(row['ISIC-Code'])))
    
    # Industry prefixes for synthetic generation
    prefixes = [
        "Manufacturing of", "Production of", "Processing of", "Services related to",
        "Wholesale of", "Retail of", "Repair of", "Maintenance of",
        "Installation of", "Construction of", "Development of", "Operations of"
    ]
    
    # Generate synthetic examples
    for _ in range(min(num_synthetic, len(activities) * 2)):
        # Pick a random activity
        activity, code = random.choice(activities)
        
        # Sometimes add prefix
        if random.random() < 0.3:
            prefix = random.choice(prefixes)
            synthetic_text = f"{prefix} {activity.lower()}"
        else:
            synthetic_text = activity
        
        synthetic_examples.append({
            'text': synthetic_text,
            'label': code,
            'original_description': activity,
            'synthetic': True
        })
    
    return synthetic_examples

def prepare_training_data(output_file: str = 'isic_training_data.json'):
    """Main function to prepare all training data"""
    print("Loading ISIC data...")
    df = load_isic_data()
    print(f"Loaded {len(df)} ISIC codes")
    
    print("Creating training examples...")
    training_examples = create_training_examples(df)
    print(f"Created {len(training_examples)} training examples")
    
    print("Creating synthetic examples...")
    synthetic_examples = create_synthetic_examples(df, num_synthetic=800)
    print(f"Created {len(synthetic_examples)} synthetic examples")
    
    # Combine all examples
    all_examples = training_examples + synthetic_examples
    random.shuffle(all_examples)
    
    # Split into train/validation
    split_idx = int(0.8 * len(all_examples))
    train_data = all_examples[:split_idx]
    val_data = all_examples[split_idx:]
    
    # Create label mapping
    unique_labels = sorted(list(set(example['label'] for example in all_examples)))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    # Add label IDs to examples
    for example in train_data:
        example['label_id'] = label_to_id[example['label']]
    for example in val_data:
        example['label_id'] = label_to_id[example['label']]
    
    # Save training data
    training_data = {
        'train': train_data,
        'validation': val_data,
        'label_to_id': label_to_id,
        'id_to_label': id_to_label,
        'num_labels': len(unique_labels),
        'stats': {
            'total_examples': len(all_examples),
            'train_examples': len(train_data),
            'val_examples': len(val_data),
            'unique_labels': len(unique_labels),
            'synthetic_examples': len(synthetic_examples)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining data saved to {output_file}")
    print(f"Statistics:")
    print(f"  Total examples: {training_data['stats']['total_examples']}")
    print(f"  Training: {training_data['stats']['train_examples']}")
    print(f"  Validation: {training_data['stats']['val_examples']}")
    print(f"  Unique ISIC codes: {training_data['stats']['unique_labels']}")
    print(f"  Synthetic examples: {training_data['stats']['synthetic_examples']}")
    
    return training_data

if __name__ == "__main__":
    prepare_training_data()