import os
import streamlit as st
import pandas as pd
import io
import time
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import concurrent.futures
import traceback
import torch
import numpy as np
import zipfile
from threading import Lock
import threading
from fine_tuned_classifier import FineTunedISICClassifier, get_classifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Qdrant client, SentenceTransformer model, and fine-tuned classifier
@st.cache_resource
def initialize_models():
    # Use GPU if available for faster encoding
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceTransformer("all-MiniLM-L12-v2", device=device)
    qdrant = QdrantClient(":memory:")
    
    # Initialize fine-tuned classifier
    fine_tuned_classifier = None
    try:
        fine_tuned_classifier = get_classifier()
        if fine_tuned_classifier:
            st.success("🤖 Fine-tuned model loaded successfully!")
        else:
            st.warning("⚠️ Fine-tuned model not found. Using embedding model only.")
    except Exception as e:
        st.warning(f"⚠️ Could not load fine-tuned model: {e}")
    
    st.write(f"Using device: {device}")
    return qdrant, encoder, fine_tuned_classifier

# Enhanced batch encoding with detailed progress display
def encode_in_batches(text_data, encoder, batch_size=256):
    vectors = []
    total_items = len(text_data)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    st.write(f"🔄 Encoding {total_items:,} ISIC definitions in {total_batches} batches...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    processed_items = 0
    
    for batch_idx, i in enumerate(range(0, len(text_data), batch_size), 1):
        batch = text_data[i:i + batch_size]
        batch_size_actual = len(batch)
        
        # Update status
        status_text.text(f"Processing batch {batch_idx}/{total_batches} ({batch_size_actual} items)...")
        
        batch_vectors = encoder.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        vectors.extend(batch_vectors.tolist())
        
        processed_items += batch_size_actual
        progress = min(processed_items / total_items, 1.0)
        progress_bar.progress(progress)
        
        # Show progress percentage
        if batch_idx % 2 == 0 or batch_idx == total_batches:  # Update every 2 batches
            elapsed = time.time() - start_time
            speed = processed_items / elapsed if elapsed > 0 else 0
            status_text.text(f"Batch {batch_idx}/{total_batches} - {progress*100:.1f}% complete - {speed:.1f} items/sec")
    
    total_time = time.time() - start_time
    status_text.text(f"✅ Encoding complete! {processed_items:,} items in {total_time:.2f}s ({processed_items/total_time:.1f} items/sec)")
    
    return vectors

# Enhanced upsert in batches to Qdrant with progress tracking
def upsert_in_batches(vectors, qdrant, data_dict, collection_name="industries", batch_size=128):
    total_vectors = len(vectors)
    total_batches = (total_vectors + batch_size - 1) // batch_size
    
    st.write(f"📤 Uploading {total_vectors:,} vectors to Qdrant in {total_batches} batches...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    processed_vectors = 0
    
    for batch_idx, i in enumerate(range(0, len(vectors), batch_size), 1):
        batch = [
            models.PointStruct(
                id=j,
                vector=vectors[j],
                payload={
                    "code": data_dict[j]['code'],
                    "title": data_dict[j]['title'],
                    "level": data_dict[j].get('level', 'Unknown')
                }
            ) for j in range(i, min(i + batch_size, len(vectors)))
        ]
        
        batch_size_actual = len(batch)
        status_text.text(f"Uploading batch {batch_idx}/{total_batches} ({batch_size_actual} vectors)...")
        
        qdrant.upsert(collection_name=collection_name, points=batch)
        
        processed_vectors += batch_size_actual
        progress = min(processed_vectors / total_vectors, 1.0)
        progress_bar.progress(progress)
        
        if batch_idx % 3 == 0 or batch_idx == total_batches:  # Update every 3 batches
            elapsed = time.time() - start_time
            speed = processed_vectors / elapsed if elapsed > 0 else 0
            status_text.text(f"Batch {batch_idx}/{total_batches} - {progress*100:.1f}% complete - {speed:.1f} vectors/sec")
    
    total_time = time.time() - start_time
    status_text.text(f"✅ Upload complete! {processed_vectors:,} vectors uploaded in {total_time:.2f}s")

# Progress bar function
def show_progress_bar(message, current, total):
    progress_value = min(current / total, 1.0)
    st.progress(progress_value)
    st.write(f"{message}: {int(progress_value * 100)}% complete.")

# Load and encode classification data (ISIC or ISCO)
@st.cache_resource
def load_and_encode_classification_data(classification_type, _qdrant, _encoder):
    start_time = time.time()
    
    # Create collection based on classification type
    collection_name = "industries" if classification_type == "ISIC" else "jobs"
    _qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=_encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    
    if classification_type == "ISIC":
        data_path = "Localised ISIC.xlsx"
        try:
            data_df = pd.read_excel(data_path, sheet_name='ISIC-Rev4')
            st.write("ISIC data loaded successfully.")
            
            # Include both Level 2 (2-digit) and Level 4 (4-digit) ISIC codes
            data_df = data_df[data_df["ISIC-Code"].astype(str).str.len().isin([2, 4])]
            data_df["text"] = data_df["ISIC-Sub Activity Description"].fillna("")

            # Add level information for clearer identification
            data_df["level"] = data_df["ISIC-Code"].astype(str).str.len().map({2: "Level 2", 4: "Level 4"})

            data_dict = data_df[["ISIC-Code", "ISIC-Sub Activity Description", "text", "level"]].rename(columns={"ISIC-Sub Activity Description": "title", "ISIC-Code": "code"}).to_dict(orient="records")

            level_2_count = len([d for d in data_dict if d["level"] == "Level 2"])
            level_4_count = len([d for d in data_dict if d["level"] == "Level 4"])
            st.write(f"Loaded {level_2_count} Level 2 (Division) and {level_4_count} Level 4 (Class) ISIC codes")
            
        except Exception as e:
            st.error(f"Error loading ISIC data: {e}")
            return None
            
    else:  # ISCO
        data_path = "isco_index.xlsx"
        try:
            data_df = pd.read_excel(data_path, sheet_name='ISCO-08 EN Struct and defin')
            st.write("ISCO data loaded successfully.")
            
            # Use Level 2 for 2-digit ISCO codes (as per original implementation)
            data_df = data_df[data_df["Level"] == 2]
            
            # Enhanced text combination for better matching (using original ISCO logic)
            data_df["text"] = data_df.apply(lambda row: " ".join([
                str(row.get("Definition", "")), 
                str(row.get("Tasks include", "")), 
                str(row.get("Included occupations", ""))
            ]), axis=1)
            
            # Add word count for analysis (optional)
            data_df["words"] = data_df.apply(lambda row: len(row["text"].split(" ")), axis=1)
            
            # Create data dictionary with proper column mapping
            data_dict = data_df[["ISCO 08 Code", "Title EN", "text"]].rename(
                columns={"Title EN": "title", "ISCO 08 Code": "code"}
            ).to_dict(orient="records")
            
            st.write(f"Loaded {len(data_dict)} ISCO occupation categories (Level 2)")
                
        except FileNotFoundError:
            st.error(f"❌ **ISCO data file not found!** Please add 'isco_index.xlsx' to the application folder.")
            st.info("💡 **Expected ISCO file structure:**\n- Sheet: 'ISCO-08 EN Struct and defin'\n- Columns: Level, ISCO 08 Code, Title EN, Definition, Tasks include, Included occupations")
            return None
        except Exception as e:
            st.error(f"Error loading ISCO data: {e}")
            st.text(traceback.format_exc())
            return None

    # Batch encode the data
    texts = [record['text'] for record in data_dict]
    vectors = encode_in_batches(texts, _encoder)

    # Upsert vectors to Qdrant in batches
    upsert_in_batches(vectors, _qdrant, data_dict, collection_name)

    end_time = time.time()
    st.write(f"{classification_type} data encoded and upserted to Qdrant. Time taken: {end_time - start_time:.2f} seconds.")
    return data_df

# Batch processing function for finding top 3 classification codes - WITH DETAILED PROGRESS TRACKING
def find_top_3_classification_codes_batch(industries, encoder, qdrant, collection_name="industries", batch_size=256):
    all_results = []
    total_items = len(industries)
    processed_count = 0
    
    st.write(f"🚀 Processing {total_items} industry descriptions with optimized batch processing...")
    
    # Create progress display elements
    progress_bar = st.progress(0)
    status_container = st.container()
    metrics_container = st.container()
    
    start_time = time.time()
    
    # Process in chunks for better memory management
    for i in range(0, len(industries), batch_size):
        batch_start_time = time.time()
        batch_industries = industries[i:i + batch_size]
        current_batch_size = len(batch_industries)
        
        # Update status before processing
        with status_container:
            st.write(f"📊 **Processing Batch {(i // batch_size) + 1} of {(total_items + batch_size - 1) // batch_size}**")
        
        # Encode the entire batch at once - major performance improvement
        encoding_start = time.time()
        batch_vectors = encoder.encode(
            batch_industries, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            batch_size=32  # Internal batch size for encoding
        )
        encoding_time = time.time() - encoding_start
        
        # Optimized vector search - process multiple vectors efficiently
        batch_results = []
        search_batch_size = 50  # Process searches in smaller batches to avoid memory issues
        
        search_start = time.time()
        for j in range(0, len(batch_vectors), search_batch_size):
            search_batch = batch_vectors[j:j + search_batch_size]
            
            # Process each vector in the search batch
            for k, vector in enumerate(search_batch):
                hits = qdrant.search(
                    collection_name=collection_name,
                    query_vector=vector.tolist(),
                    limit=6  # Get more results to ensure we have both levels
                )

                # Separate level 2 and level 4 codes
                level_2_codes = []
                level_4_codes = []

                for hit in hits:
                    code = hit.payload.get('code')
                    level = hit.payload.get('level', 'Unknown')

                    if level == 'Level 2' and len(level_2_codes) < 3:
                        level_2_codes.append(code)
                    elif level == 'Level 4' and len(level_4_codes) < 3:
                        level_4_codes.append(code)

                # Pad with None to ensure 3 results each
                level_2_codes += [None] * (3 - len(level_2_codes))
                level_4_codes += [None] * (3 - len(level_4_codes))

                # Combine results: [level2_1, level2_2, level2_3, level4_1, level4_2, level4_3]
                combined_results = level_2_codes + level_4_codes
                batch_results.append(combined_results)
                
                # Update processed count and progress for individual items within batch
                processed_count += 1
                
                # Update progress bar and metrics more frequently for better UX
                if k % 5 == 0 or processed_count == total_items:  # Update every 5 items or at the end
                    progress_percentage = (processed_count / total_items) * 100
                    progress_bar.progress(processed_count / total_items)
                    
                    # Calculate metrics
                    elapsed_time = time.time() - start_time
                    items_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                    eta_seconds = (total_items - processed_count) / items_per_second if items_per_second > 0 else 0
                    
                    with metrics_container:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Progress", 
                                f"{progress_percentage:.1f}%", 
                                f"{processed_count:,} / {total_items:,} records"
                            )
                        with col2:
                            st.metric(
                                "Processing Speed", 
                                f"{items_per_second:.1f}/sec",
                                f"Elapsed: {elapsed_time:.1f}s"
                            )
                        with col3:
                            if eta_seconds > 0:
                                eta_minutes = eta_seconds / 60
                                if eta_minutes > 1:
                                    eta_display = f"{eta_minutes:.1f}m"
                                else:
                                    eta_display = f"{eta_seconds:.0f}s"
                                st.metric(
                                    "ETA", 
                                    eta_display,
                                    f"Remaining: {total_items - processed_count:,}"
                                )
                            else:
                                st.metric("ETA", "Calculating...", "")
        
        search_time = time.time() - search_start
        batch_time = time.time() - batch_start_time
        
        all_results.extend(batch_results)
        
        # Update detailed status after batch completion
        with status_container:
            st.write(f"✅ **Batch {(i // batch_size) + 1} Complete** - "
                    f"Encoding: {encoding_time:.2f}s, Search: {search_time:.2f}s, "
                    f"Total: {batch_time:.2f}s ({current_batch_size} items)")
    
    # Final status update
    total_time = time.time() - start_time
    final_speed = total_items / total_time if total_time > 0 else 0
    
    progress_bar.progress(1.0)
    with status_container:
        st.success(f"🎉 **Processing Complete!** "
                  f"Processed {processed_count:,} records in {total_time:.2f}s "
                  f"(Average: {final_speed:.1f} items/sec)")
    
    # Keep progress elements visible for a moment before clearing
    time.sleep(2)
    progress_bar.empty()
    
    return all_results

# Smart batching logic for multiple files
def calculate_file_processing_priority(uploaded_files):
    """
    Calculate processing priority based on file size and complexity
    Returns: (small_files, medium_files, large_files)
    """
    small_files = []   # < 1MB or < 1000 records
    medium_files = []  # 1MB-10MB or 1000-10000 records
    large_files = []   # > 10MB or > 10000 records
    
    for file in uploaded_files:
        file_size_mb = file.size / (1024 * 1024)
        
        # Quick record count estimation (rough)
        if file.name.endswith('.csv'):
            # Estimate records based on file size (rough)
            estimated_records = file_size_mb * 1000  # rough estimate
        else:  # Excel
            estimated_records = file_size_mb * 500   # Excel is typically more compact
        
        if file_size_mb < 1 or estimated_records < 1000:
            small_files.append((file, estimated_records))
        elif file_size_mb < 10 or estimated_records < 10000:
            medium_files.append((file, estimated_records))
        else:
            large_files.append((file, estimated_records))
    
    return small_files, medium_files, large_files

# Thread-safe progress tracker
class MultiFileProgressTracker:
    def __init__(self, total_files):
        self.total_files = total_files
        self.completed_files = 0
        self.file_progress = {}
        self.file_status = {}
        self.lock = Lock()
        
    def update_file_progress(self, filename, progress_pct, status="Processing"):
        with self.lock:
            self.file_progress[filename] = progress_pct
            self.file_status[filename] = status
            
    def complete_file(self, filename, success=True):
        with self.lock:
            self.completed_files += 1
            self.file_progress[filename] = 100
            self.file_status[filename] = "✅ Complete" if success else "❌ Failed"
    
    def get_overall_progress(self):
        with self.lock:
            if self.total_files == 0:
                return 0
            return (self.completed_files / self.total_files) * 100

# Enhanced concurrent file processor
def process_multiple_files_hybrid(uploaded_files, encoder, qdrant, classification_dir, classification_type, progress_tracker, fine_tuned_classifier=None, use_fine_tuned=False):
    """
    Hybrid approach: Smart batching + concurrent processing
    """
    small_files, medium_files, large_files = calculate_file_processing_priority(uploaded_files)
    
    st.write(f"📊 **File Analysis**: {len(small_files)} small, {len(medium_files)} medium, {len(large_files)} large files")
    
    all_results = []
    
    # Process large files first, one at a time (to avoid memory issues)
    if large_files:
        st.write("🔄 **Processing large files sequentially...**")
        for file_info in large_files:
            file, estimated_records = file_info
            progress_tracker.update_file_progress(file.name, 0, "Starting large file")
            try:
                if use_fine_tuned and fine_tuned_classifier:
                    result = process_file_with_fine_tuned(file, fine_tuned_classifier, classification_dir, classification_type)
                else:
                    result = process_file_compact(file, encoder, qdrant, classification_dir, classification_type)
                result['estimated_records'] = estimated_records
                all_results.append(result)
                progress_tracker.complete_file(file.name, True)
            except Exception as e:
                st.error(f"❌ Error processing large file {file.name}: {e}")
                progress_tracker.complete_file(file.name, False)
                all_results.append({'filename': file.name, 'error': str(e)})
    
    # Process medium files with limited concurrency
    if medium_files:
        st.write("⚡ **Processing medium files with limited concurrency...**")
        max_concurrent_medium = min(3, len(medium_files))  # Max 3 concurrent medium files
        
        def process_medium_file(file_info):
            file, estimated_records = file_info
            progress_tracker.update_file_progress(file.name, 0, "Processing medium file")
            try:
                if use_fine_tuned and fine_tuned_classifier:
                    result = process_file_with_fine_tuned(file, fine_tuned_classifier, classification_dir, classification_type)
                else:
                    result = process_file_compact(file, encoder, qdrant, classification_dir, classification_type)
                result['estimated_records'] = estimated_records
                progress_tracker.complete_file(file.name, True)
                return result
            except Exception as e:
                progress_tracker.complete_file(file.name, False)
                return {'filename': file.name, 'error': str(e)}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_medium) as executor:
            medium_results = list(executor.map(process_medium_file, medium_files))
            all_results.extend(medium_results)
    
    # Process small files with high concurrency
    if small_files:
        st.write("🚀 **Processing small files with high concurrency...**")
        max_concurrent_small = min(5, len(small_files))  # Max 5 concurrent small files
        
        def process_small_file(file_info):
            file, estimated_records = file_info
            progress_tracker.update_file_progress(file.name, 0, "Processing small file")
            try:
                if use_fine_tuned and fine_tuned_classifier:
                    result = process_file_with_fine_tuned(file, fine_tuned_classifier, classification_dir, classification_type)
                else:
                    result = process_file_compact(file, encoder, qdrant, classification_dir, classification_type)
                result['estimated_records'] = estimated_records
                progress_tracker.complete_file(file.name, True)
                return result
            except Exception as e:
                progress_tracker.complete_file(file.name, False)
                return {'filename': file.name, 'error': str(e)}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_small) as executor:
            small_results = list(executor.map(process_small_file, small_files))
            all_results.extend(small_results)
    
    return all_results

# Process file using fine-tuned model
def process_file_with_fine_tuned(uploaded_file, fine_tuned_classifier, save_directory, classification_type="ISIC"):
    """Process file using fine-tuned DistilBERT model"""
    start_time = time.time()
    
    # Load file
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    
    total_records = len(df)
    
    # Check for required column
    required_col = 'INDUSTRY'
    if required_col not in df.columns:
        st.error(f"❌ Missing '{required_col}' column. Available: {', '.join(df.columns.tolist())}")
        raise ValueError(f"Missing {required_col} column")
    
    # Data preparation
    df['combined_text'] = df[required_col].astype(str)
    
    # Filter valid rows
    valid_mask = df['combined_text'].str.len() > 2
    valid_industries = df.loc[valid_mask, 'combined_text'].tolist()
    filtered_count = len(valid_industries)
    
    if filtered_count < total_records:
        skipped_count = total_records - filtered_count
        st.warning(f"⚠️ Skipped {skipped_count} rows with empty/short INDUSTRY values. Processing {filtered_count:,} valid records.")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process with fine-tuned model
    st.info("🤖 Using fine-tuned DistilBERT model for classification")
    
    # Batch process with fine-tuned model
    batch_results = []
    batch_size = 32  # Smaller batches for fine-tuned model
    
    for i in range(0, len(valid_industries), batch_size):
        batch = valid_industries[i:i + batch_size]
        
        # Get predictions for batch
        batch_predictions = []
        for text in batch:
            predictions = fine_tuned_classifier.predict_single(text, top_k=6)  # Get more predictions

            # Extract level 4 codes and derive level 2 codes
            level_4_codes = [pred['code'] for pred in predictions if len(pred['code']) == 4][:3]
            level_2_codes = list(dict.fromkeys([code[:2] for code in level_4_codes if len(code) == 4]))[:3]  # Remove duplicates, keep order

            # Pad with None to ensure 3 results each
            level_2_codes += [None] * (3 - len(level_2_codes))
            level_4_codes += [None] * (3 - len(level_4_codes))

            # Combine results: [level2_1, level2_2, level2_3, level4_1, level4_2, level4_3]
            combined_codes = level_2_codes + level_4_codes
            batch_predictions.append(combined_codes)
        
        batch_results.extend(batch_predictions)
        
        # Update progress
        progress = (i + len(batch)) / len(valid_industries)
        progress_bar.progress(progress)
        status_text.text(f"Fine-tuned model processing: {i + len(batch):,}/{len(valid_industries):,} ({progress*100:.0f}%)")
    
    # Create full results matching original dataframe
    full_results = []
    batch_idx = 0
    
    for idx in df.index:
        if valid_mask.iloc[idx]:
            full_results.append(batch_results[batch_idx])
            batch_idx += 1
        else:
            full_results.append([None, None, None, None, None, None])  # 6 columns for both levels

    # Add results to dataframe with both level 2 and level 4 classifications
    results_df = pd.DataFrame(full_results, columns=[
        'isic_level2_code_1', 'isic_level2_code_2', 'isic_level2_code_3',
        'isic_level4_code_1', 'isic_level4_code_2', 'isic_level4_code_3'
    ])
    df[['isic_level2_code_1', 'isic_level2_code_2', 'isic_level2_code_3',
        'isic_level4_code_1', 'isic_level4_code_2', 'isic_level4_code_3']] = results_df
    
    # Save file
    name_parts = uploaded_file.name.rsplit('.', 1)
    if len(name_parts) == 2:
        base_name, original_ext = name_parts
        processed_filename = f"{base_name}_fine_tuned.{original_ext}"
    else:
        processed_filename = f"{uploaded_file.name}_fine_tuned.csv"
    
    save_path = os.path.join(save_directory, processed_filename)
    
    # Prepare file data for download
    file_data = None
    mime_type = None
    
    try:
        if processed_filename.endswith('.xlsx'):
            df.to_excel(save_path, index=False, engine='openpyxl')
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            file_data = excel_buffer.getvalue()
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            file_data = csv_buffer.getvalue().encode('utf-8-sig')
            mime_type = 'text/csv'
    except Exception as e:
        # Fallback to CSV
        fallback_path = save_path.replace('.xlsx', '.csv') if save_path.endswith('.xlsx') else save_path
        processed_filename = processed_filename.replace('.xlsx', '.csv') if processed_filename.endswith('.xlsx') else processed_filename
        df.to_csv(fallback_path, index=False, encoding='utf-8-sig')
        save_path = fallback_path
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        file_data = csv_buffer.getvalue().encode('utf-8-sig')
        mime_type = 'text/csv'
    
    end_time = time.time()
    total_time = end_time - start_time
    speed = filtered_count / total_time if total_time > 0 else 0
    
    progress_bar.progress(1.0)
    status_text.text(f"✅ Fine-tuned model processing complete: {filtered_count:,} records in {total_time:.1f}s ({speed:.0f}/sec)")
    
    return {
        'records': filtered_count,
        'time': total_time,
        'speed': speed,
        'path': save_path,
        'filename': processed_filename,
        'file_data': file_data,
        'mime_type': mime_type
    }

# Accuracy testing function
def run_accuracy_test(test_file, encoder, qdrant, fine_tuned_classifier, classification_type):
    """Run accuracy test comparing embedding vs fine-tuned model"""
    try:
        # Load test data
        if test_file.name.endswith('.xlsx'):
            test_df = pd.read_excel(test_file)
        else:
            test_df = pd.read_csv(test_file)
        
        st.info(f"📊 Loaded {len(test_df)} test records")
        
        # Check required columns
        required_cols = ['INDUSTRY']
        expected_col = None
        
        # Look for expected ISIC column
        possible_expected_cols = ['ISIC', 'ISIC_CODE', 'EXPECTED_ISIC', 'TRUE_ISIC', 'ACTUAL_ISIC']
        for col in possible_expected_cols:
            if col in test_df.columns:
                expected_col = col
                break
        
        if not expected_col:
            st.error(f"❌ No expected ISIC column found. Looking for one of: {', '.join(possible_expected_cols)}")
            st.info(f"Available columns: {', '.join(test_df.columns.tolist())}")
            return
        
        if 'INDUSTRY' not in test_df.columns:
            st.error("❌ Missing 'INDUSTRY' column")
            return
        
        # Filter valid test data
        test_df = test_df.dropna(subset=['INDUSTRY', expected_col])
        test_df = test_df[test_df['INDUSTRY'].str.len() > 2]
        test_df = test_df[test_df[expected_col].astype(str).str.len().isin([2, 4])]  # Both level 2 and 4 ISIC codes
        
        if len(test_df) == 0:
            st.error("❌ No valid test data found after filtering")
            return
        
        st.success(f"✅ Using {len(test_df)} valid test records")
        
        # Prepare test data
        test_industries = test_df['INDUSTRY'].tolist()
        true_codes = test_df[expected_col].astype(str).tolist()
        
        # Test embedding model
        st.markdown("### 🔬 Testing Embedding Model")
        embedding_progress = st.progress(0)
        embedding_status = st.empty()
        
        embedding_predictions = []
        for i, industry in enumerate(test_industries):
            vector = encoder.encode(industry).tolist()
            hits = qdrant.search(collection_name="industries", query_vector=vector, limit=1)
            pred_code = hits[0].payload.get('code') if hits else None
            embedding_predictions.append(pred_code)
            
            if i % 10 == 0 or i == len(test_industries) - 1:
                progress = (i + 1) / len(test_industries)
                embedding_progress.progress(progress)
                embedding_status.text(f"Embedding model: {i+1}/{len(test_industries)} ({progress*100:.0f}%)")
        
        # Test fine-tuned model
        st.markdown("### 🤖 Testing Fine-tuned Model")
        finetuned_progress = st.progress(0)
        finetuned_status = st.empty()
        
        finetuned_predictions = []
        for i, industry in enumerate(test_industries):
            predictions = fine_tuned_classifier.predict_single(industry, top_k=1)
            pred_code = predictions[0]['code'] if predictions else None
            finetuned_predictions.append(pred_code)
            
            if i % 10 == 0 or i == len(test_industries) - 1:
                progress = (i + 1) / len(test_industries)
                finetuned_progress.progress(progress)
                finetuned_status.text(f"Fine-tuned model: {i+1}/{len(test_industries)} ({progress*100:.0f}%)")
        
        # Calculate accuracies
        embedding_accuracy = accuracy_score(true_codes, embedding_predictions)
        finetuned_accuracy = accuracy_score(true_codes, finetuned_predictions)
        
        # Display results
        st.markdown("### 📈 Accuracy Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Embedding Model", f"{embedding_accuracy:.1%}", f"{embedding_accuracy:.3f}")
        with col2:
            st.metric("Fine-tuned Model", f"{finetuned_accuracy:.1%}", f"{finetuned_accuracy:.3f}")
        with col3:
            improvement = finetuned_accuracy - embedding_accuracy
            st.metric("Improvement", f"{improvement:+.1%}", f"{improvement:+.3f}")
        
        # Detailed analysis
        with st.expander("📊 Detailed Analysis"):
            # Create comparison dataframe
            results_df = pd.DataFrame({
                'Industry': test_industries,
                'True_ISIC': true_codes,
                'Embedding_Pred': embedding_predictions,
                'FineTuned_Pred': finetuned_predictions,
                'Embedding_Correct': [str(t) == str(p) for t, p in zip(true_codes, embedding_predictions)],
                'FineTuned_Correct': [str(t) == str(p) for t, p in zip(true_codes, finetuned_predictions)]
            })
            
            # Show sample results
            st.markdown("**Sample Results:**")
            st.dataframe(results_df.head(10))
            
            # Show cases where fine-tuned is better
            better_cases = results_df[
                (~results_df['Embedding_Correct']) & 
                (results_df['FineTuned_Correct'])
            ]
            
            if len(better_cases) > 0:
                st.markdown(f"**Cases where fine-tuned model performs better ({len(better_cases)} cases):**")
                st.dataframe(better_cases[['Industry', 'True_ISIC', 'Embedding_Pred', 'FineTuned_Pred']].head(5))
            
            # Classification reports
            st.markdown("**Embedding Model Classification Report:**")
            st.text(classification_report(true_codes, embedding_predictions, zero_division=0))
            
            st.markdown("**Fine-tuned Model Classification Report:**")
            st.text(classification_report(true_codes, finetuned_predictions, zero_division=0))
        
        # Save results
        results_df['Test_File'] = test_file.name
        results_df['Test_Date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare download
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download Test Results",
            data=csv_data,
            file_name=f"accuracy_test_results_{int(time.time())}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"❌ Error running accuracy test: {e}")
        with st.expander("🔍 Error Details"):
            st.text(traceback.format_exc())

# Create bulk download functionality
def create_bulk_download_zip(results, classification_type):
    """
    Create a ZIP file containing all processed files
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for result in results:
            if 'file_data' in result and result['file_data']:
                zip_file.writestr(result['filename'], result['file_data'])
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# Keep the original function for backward compatibility (not used in optimized version)
def find_top_3_classification_codes(industry, encoder, qdrant, collection_name="industries"):
    vector = encoder.encode(industry).tolist()
    hits = qdrant.search(collection_name=collection_name, query_vector=vector, limit=3)
    results = [hit.payload.get('code') for hit in hits]
    return results + [None] * (3 - len(results))

# Compact progress function for better UX
def process_file_compact(uploaded_file, encoder, qdrant, save_directory, classification_type="ISIC"):
    """
    Compact version of file processing with minimal UI elements
    Returns processing results for history tracking
    """
    start_time = time.time()
    
    # Quick file loading
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    
    total_records = len(df)
    
    # Check for required columns based on classification type
    if classification_type == "ISIC":
        required_col = 'INDUSTRY'
        if required_col not in df.columns:
            st.error(f"❌ **Missing '{required_col}' column for ISIC classification.** Available: {', '.join(df.columns.tolist())}")
            raise ValueError(f"Missing {required_col} column")
    else:  # ISCO
        # For ISCO, we can use either OCCUPATION or INDUSTRY column
        if 'OCCUPATION' in df.columns:
            required_col = 'OCCUPATION'
        elif 'INDUSTRY' in df.columns:
            required_col = 'INDUSTRY'
        else:
            st.error(f"❌ **Missing 'OCCUPATION' or 'INDUSTRY' column for ISCO classification.** Available: {', '.join(df.columns.tolist())}")
            raise ValueError("Missing OCCUPATION or INDUSTRY column")
    
    # Data preparation with better empty value handling
    # Check if classification-specific columns exist
    has_class_col = 'isic_class' in df.columns or 'isco_class' in df.columns
    
    # Create a mask for valid rows
    def is_valid_row(row):
        # Check main column (INDUSTRY/OCCUPATION)
        main_text = str(row[required_col]).strip()
        main_valid = len(main_text) > 2 and main_text.lower() not in ['nan', 'none', '']
        
        # If classification-specific columns exist, check them too
        if has_class_col:
            class_col = 'isic_class' if 'isic_class' in df.columns else 'isco_class'
            class_text = str(row[class_col]).strip()
            class_valid = len(class_text) > 0 and class_text.lower() not in ['nan', 'none', '']
            return main_valid and class_valid
        else:
            return main_valid
    
    # Apply validation and create processing lists
    df['is_valid_for_processing'] = df.apply(is_valid_row, axis=1)
    
    # Create combined text for processing based on classification type
    if classification_type == "ISCO" and 'DESCRIPTION' in df.columns:
        # For ISCO, combine OCCUPATION and DESCRIPTION if both are available
        df['combined_text'] = df[[required_col, 'DESCRIPTION']].astype(str).agg(' '.join, axis=1)
        st.info(f"🔗 Using {required_col} + DESCRIPTION for enhanced ISCO matching")
    else:
        # For ISIC or ISCO without description, use main column only
        df['combined_text'] = df[required_col].astype(str)
        st.info(f"📝 Using {required_col} column for {classification_type} classification")
    
    # Only process valid rows
    valid_indices = df[df['is_valid_for_processing']].index.tolist()
    industries_list = [df.loc[idx, 'combined_text'] for idx in valid_indices]
    filtered_count = len(industries_list)
    
    if filtered_count < total_records:
        skipped_count = total_records - filtered_count
        class_info = f" or {classification_type.lower()}_class" if has_class_col else ""
        st.warning(f"⚠️ Skipped {skipped_count} rows with empty {required_col}{class_info} values. Processing {filtered_count:,} valid records.")
    
    # Compact progress display
    progress_col1, progress_col2 = st.columns([3, 1])
    
    with progress_col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    with progress_col2:
        metrics_container = st.empty()
    
    # Process with compact progress tracking
    collection_name = "industries" if classification_type == "ISIC" else "jobs"
    batch_results = find_top_3_classification_codes_compact(industries_list, encoder, qdrant, progress_bar, status_text, metrics_container, collection_name)
    
    # Quick results assembly with improved validation handling
    if filtered_count < total_records:
        # Create results array matching original dataframe length
        full_results = []
        batch_idx = 0
        
        for idx in df.index:
            if df.loc[idx, 'is_valid_for_processing']:
                # Use classification results for valid rows
                full_results.append(batch_results[batch_idx])
                batch_idx += 1
            else:
                # Skip classification for invalid rows (empty INDUSTRY or isic_class)
                if classification_type == "ISIC":
                    full_results.append([None, None, None, None, None, None])  # 6 columns for both levels
                else:
                    full_results.append([None, None, None])  # ISCO remains 3 columns

        if classification_type == "ISIC":
            # ISIC with both level 2 and level 4 classifications
            results_df = pd.DataFrame(full_results, columns=[
                'isic_level2_code_1', 'isic_level2_code_2', 'isic_level2_code_3',
                'isic_level4_code_1', 'isic_level4_code_2', 'isic_level4_code_3'
            ])
        else:
            # ISCO remains unchanged
            results_df = pd.DataFrame(full_results, columns=[f'{classification_type.lower()}_code_1', f'{classification_type.lower()}_code_2', f'{classification_type.lower()}_code_3'])
    else:
        if classification_type == "ISIC":
            # ISIC with both level 2 and level 4 classifications
            results_df = pd.DataFrame(batch_results, columns=[
                'isic_level2_code_1', 'isic_level2_code_2', 'isic_level2_code_3',
                'isic_level4_code_1', 'isic_level4_code_2', 'isic_level4_code_3'
            ])
        else:
            # ISCO remains unchanged
            results_df = pd.DataFrame(batch_results, columns=[f'{classification_type.lower()}_code_1', f'{classification_type.lower()}_code_2', f'{classification_type.lower()}_code_3'])
    
    # Add classification codes to the dataframe
    if classification_type == "ISIC":
        df[['isic_level2_code_1', 'isic_level2_code_2', 'isic_level2_code_3',
            'isic_level4_code_1', 'isic_level4_code_2', 'isic_level4_code_3']] = results_df
    else:
        df[[f'{classification_type.lower()}_code_1', f'{classification_type.lower()}_code_2', f'{classification_type.lower()}_code_3']] = results_df
    
    # Remove temporary columns before saving
    df_output = df.drop(columns=['combined_text', 'is_valid_for_processing'], errors='ignore')
    
    # Reorder columns to put classification codes after the original columns
    if classification_type == "ISIC":
        # For ISIC, we have both level 2 and level 4 codes
        code_prefixes = ['isic_level2_code', 'isic_level4_code']
        original_cols = [col for col in df_output.columns if not any(col.startswith(prefix) for prefix in code_prefixes)]
        code_cols = [col for col in df_output.columns if any(col.startswith(prefix) for prefix in code_prefixes)]
    else:
        # For ISCO, keep existing logic
        code_prefix = f'{classification_type.lower()}_code'
        original_cols = [col for col in df_output.columns if not col.startswith(code_prefix)]
        code_cols = [col for col in df_output.columns if col.startswith(code_prefix)]

    df_output = df_output[original_cols + code_cols]
    
    # Create processed filename with proper extension
    name_parts = uploaded_file.name.rsplit('.', 1)
    if len(name_parts) == 2:
        base_name, original_ext = name_parts
        processed_filename = f"{base_name}_processed.{original_ext}"
    else:
        processed_filename = f"{uploaded_file.name}_processed.csv"
    
    save_path = os.path.join(save_directory, processed_filename)
    
    # Save in original format when possible AND prepare file data for download
    file_data = None
    mime_type = None
    
    try:
        if processed_filename.endswith('.xlsx'):
            # Save as Excel if original was Excel
            df_output.to_excel(save_path, index=False, engine='openpyxl')
            # Prepare Excel data for download
            excel_buffer = io.BytesIO()
            df_output.to_excel(excel_buffer, index=False, engine='openpyxl')
            file_data = excel_buffer.getvalue()
            mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        else:
            # Save as CSV for all other cases
            df_output.to_csv(save_path, index=False, encoding='utf-8-sig')  # UTF-8 BOM for Excel compatibility
            # Prepare CSV data for download
            csv_buffer = io.StringIO()
            df_output.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            file_data = csv_buffer.getvalue().encode('utf-8-sig')
            mime_type = 'text/csv'
    except Exception as e:
        # Fallback to CSV if Excel save fails
        fallback_path = save_path.replace('.xlsx', '.csv') if save_path.endswith('.xlsx') else save_path
        processed_filename = processed_filename.replace('.xlsx', '.csv') if processed_filename.endswith('.xlsx') else processed_filename
        df_output.to_csv(fallback_path, index=False, encoding='utf-8-sig')
        save_path = fallback_path
        # Prepare CSV data for download
        csv_buffer = io.StringIO()
        df_output.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        file_data = csv_buffer.getvalue().encode('utf-8-sig')
        mime_type = 'text/csv'
    
    end_time = time.time()
    total_time = end_time - start_time
    speed = filtered_count / total_time if total_time > 0 else 0
    
    return {
        'records': filtered_count,
        'time': total_time,
        'speed': speed,
        'path': save_path,
        'filename': processed_filename,
        'file_data': file_data,
        'mime_type': mime_type
    }
# Compact batch processing for minimal UI
def find_top_3_classification_codes_compact(industries, encoder, qdrant, progress_bar, status_text, metrics_container, collection_name="industries", batch_size=256):
    """Compact version with minimal UI updates"""
    all_results = []
    total_items = len(industries)
    processed_count = 0
    start_time = time.time()
    
    for i in range(0, len(industries), batch_size):
        batch_industries = industries[i:i + batch_size]
        
        # Encode batch
        batch_vectors = encoder.encode(
            batch_industries, 
            show_progress_bar=False, 
            convert_to_numpy=True,
            batch_size=32
        )
        
        # Search for each vector
        batch_results = []
        for vector in batch_vectors:
            hits = qdrant.search(
                collection_name=collection_name,
                query_vector=vector.tolist(),
                limit=6 if collection_name == "industries" else 3  # Get more results for ISIC to ensure both levels
            )

            if collection_name == "industries":  # ISIC classification
                # Separate level 2 and level 4 codes
                level_2_codes = []
                level_4_codes = []

                for hit in hits:
                    code = hit.payload.get('code')
                    level = hit.payload.get('level', 'Unknown')

                    if level == 'Level 2' and len(level_2_codes) < 3:
                        level_2_codes.append(code)
                    elif level == 'Level 4' and len(level_4_codes) < 3:
                        level_4_codes.append(code)

                # Pad with None to ensure 3 results each
                level_2_codes += [None] * (3 - len(level_2_codes))
                level_4_codes += [None] * (3 - len(level_4_codes))

                # Combine results: [level2_1, level2_2, level2_3, level4_1, level4_2, level4_3]
                combined_results = level_2_codes + level_4_codes
                batch_results.append(combined_results)
            else:  # ISCO classification
                codes = [hit.payload.get('code') for hit in hits]
                batch_results.append(codes + [None] * (3 - len(codes)))
            
            processed_count += 1
            
            # Update progress every 10 items for performance
            if processed_count % 10 == 0 or processed_count == total_items:
                progress = processed_count / total_items
                progress_bar.progress(progress)
                
                elapsed = time.time() - start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = total_items - processed_count
                eta = remaining / speed if speed > 0 else 0
                
                status_text.text(f"Processing: {processed_count:,}/{total_items:,} ({progress*100:.0f}%)")
                
                with metrics_container.container():
                    st.metric("Speed", f"{speed:.0f}/sec")
                    if eta > 0 and eta < 300:  # Show ETA only if reasonable
                        st.metric("ETA", f"{eta:.0f}s")
        
        all_results.extend(batch_results)
    
    # Final update
    progress_bar.progress(1.0)
    total_time = time.time() - start_time
    final_speed = total_items / total_time if total_time > 0 else 0
    status_text.text(f"✅ Complete: {processed_count:,} records in {total_time:.1f}s ({final_speed:.0f}/sec)")
    
    return all_results

# Process individual file - ENHANCED with detailed progress tracking (Legacy function)
def process_file(uploaded_file, encoder, qdrant, save_directory, progress_container):
    try:
        # Immediate feedback when file processing starts
        st.info(f"🚀 **Starting to process:** {uploaded_file.name}")
        
        # File loading phase
        file_loading_start = time.time()
        loading_status = st.empty()
        loading_status.info(f"📁 **Loading file:** {uploaded_file.name}")
        
        # Add a small delay to make the loading status visible
        time.sleep(0.5)
        
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        file_loading_time = time.time() - file_loading_start
        
        # Update loading status to show completion
        loading_status.success(f"📊 **File loaded successfully!** Loading time: {file_loading_time:.2f}s")
        
        # File overview
        total_records = len(df)
        st.info(f"📋 **File contains:** {total_records:,} records")

        start_time = time.time()

        if 'INDUSTRY' in df.columns:
            # Data preparation phase
            st.info("🔄 **Preparing data for processing...**")
            prep_start = time.time()
            
            df['combined_text'] = df['INDUSTRY'].astype(str)
            industries_list = df['combined_text'].tolist()
            
            # Remove empty or very short entries
            original_count = len(industries_list)
            industries_list = [text for text in industries_list if len(str(text).strip()) > 2]
            filtered_count = len(industries_list)
            
            prep_time = time.time() - prep_start
            
            if filtered_count < original_count:
                st.warning(f"⚠️ Filtered out {original_count - filtered_count} empty/invalid entries. Processing {filtered_count:,} valid records.")
            
            st.success(f"✅ **Data preparation complete** ({prep_time:.2f}s)")
            
            # Create file progress header
            st.markdown("---")
            st.markdown(f"### 🚀 Processing File: `{uploaded_file.name}`")
            
            # File summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{filtered_count:,}")
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.name.split('.')[-1].upper())
            with col4:
                st.metric("Status", "Processing...")
            
            # Use enhanced batch processing with progress tracking
            batch_results = find_top_3_isic_codes_batch(industries_list, encoder, qdrant)
            
            # Results assembly phase
            st.info("📋 **Assembling results...**")
            assembly_start = time.time()
            
            # Handle the case where some records were filtered out
            if filtered_count < original_count:
                # Create full results array matching original dataframe
                full_results = []
                filtered_idx = 0
                
                for i, original_text in enumerate(df['combined_text'].astype(str)):
                    if len(original_text.strip()) > 2:
                        full_results.append(batch_results[filtered_idx])
                        filtered_idx += 1
                    else:
                        full_results.append([None, None, None])  # Empty results for filtered entries
                
                results_df = pd.DataFrame(full_results, columns=['isic_code_1', 'isic_code_2', 'isic_code_3'])
            else:
                results_df = pd.DataFrame(batch_results, columns=['isic_code_1', 'isic_code_2', 'isic_code_3'])
            
            df[['isic_code_1', 'isic_code_2', 'isic_code_3']] = results_df
            
            assembly_time = time.time() - assembly_start
            
            # File saving phase
            st.info("💾 **Saving processed file...**")
            save_start = time.time()
            
            save_path = os.path.join(save_directory, uploaded_file.name)
            df.to_csv(save_path, index=False)
            
            save_time = time.time() - save_start
            
            # Final summary
            end_time = time.time()
            total_processing_time = end_time - start_time
            items_per_second = filtered_count / total_processing_time if total_processing_time > 0 else 0
            
            st.markdown("---")
            st.success(f"🎉 **File Processing Complete!**")
            
            # Final metrics
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            with summary_col1:
                st.metric("Records Processed", f"{filtered_count:,}", f"{filtered_count}")
            with summary_col2:
                st.metric("Processing Speed", f"{items_per_second:.1f}/sec", f"{items_per_second:.1f}")
            with summary_col3:
                st.metric("Total Time", f"{total_processing_time:.2f}s", f"{total_processing_time:.2f}")
            with summary_col4:
                st.metric("Success Rate", "100%", "All records processed")
            
            # Detailed timing breakdown
            with st.expander("📈 Detailed Performance Breakdown"):
                st.write(f"• **File Loading:** {file_loading_time:.2f}s")
                st.write(f"• **Data Preparation:** {prep_time:.2f}s")
                st.write(f"• **ISIC Code Processing:** {total_processing_time - prep_time - assembly_time - save_time:.2f}s")
                st.write(f"• **Results Assembly:** {assembly_time:.2f}s")
                st.write(f"• **File Saving:** {save_time:.2f}s")
                st.write(f"• **Total Processing Time:** {total_processing_time:.2f}s")
            
            st.info(f"💾 **File saved to:** `{save_path}`")
            
        else:
            st.error(f"❌ **Error:** The uploaded file '{uploaded_file.name}' must contain an 'INDUSTRY' column.")
            st.info("📋 **Available columns:** " + ", ".join(df.columns.tolist()))
    except Exception as e:
        st.error(f"Error processing file '{uploaded_file.name}': {e}")
        st.text(traceback.format_exc())

# Sidebar configuration
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Processing settings
    st.markdown("**Processing Options:**")
    batch_size = st.select_slider("Batch Size", options=[128, 256, 512], value=256, help="Larger batches = faster processing but more memory")
    show_detailed_progress = st.checkbox("Show Detailed Progress", value=False, help="Show detailed batch information (slower)")
    
    # File management
    st.markdown("**File Management:**")
    if st.button("🗑️ Clear Processing History"):
        if 'processed_files' in st.session_state:
            st.session_state.processed_files = []
            st.success("History cleared!")
            time.sleep(1)
            st.rerun()
    
    # Statistics
    if 'processed_files' in st.session_state and st.session_state.processed_files:
        st.markdown("**Session Statistics:**")
        total_records = sum(f['records'] for f in st.session_state.processed_files)
        total_time = sum(f['time'] for f in st.session_state.processed_files)
        avg_speed = total_records / total_time if total_time > 0 else 0
        
        st.metric("Files Processed", len(st.session_state.processed_files))
        st.metric("Total Records", f"{total_records:,}")
        st.metric("Average Speed", f"{avg_speed:.0f}/sec")
    
    # Help section
    with st.expander("❓ Help"):
        st.markdown(
            "**How to use:**\\n"
            "1. Enter your name\\n"
            "2. Upload CSV/Excel file with 'INDUSTRY' column\\n"
            "3. Watch real-time progress\\n"
            "4. Download results and upload next file\\n\\n"
            "**Tips:**\\n"
            "• Larger batch sizes = faster processing\\n"
            "• CSV files process faster than Excel\\n"
            "• Clean data = better results"
        )

# Main application header
st.title("🏭 Industry Classification Mapper")
st.markdown("**AI-powered industry classification with ISIC and ISCO codes**")

# Step 1: Classification type and model selection
st.markdown("### 🎯 Select Classification Type")
classification_type = st.radio(
    "Choose the classification system you want to use:",
    ("ISIC (International Standard Industrial Classification)", "ISCO (International Standard Classification of Occupations)"),
    help="ISIC classifies economic activities/industries, while ISCO classifies occupations/jobs"
)

# Extract the classification type for easier handling
selected_classification = "ISIC" if "ISIC" in classification_type else "ISCO"

# Update title based on selection
if selected_classification == "ISIC":
    st.info("🏭 **ISIC Classification Selected** - Mapping industries to economic activity codes")
else:
    st.info("👨‍💼 **ISCO Classification Selected** - Mapping to occupational codes")

# Step 2: User input for name
user_name = st.text_input("Please enter your name:")

if user_name:
    outputs_dir = "outputs"
    user_dir = os.path.join(outputs_dir, user_name)
    
    # Create classification-specific directories
    isic_dir = os.path.join(user_dir, "ISIC")
    isco_dir = os.path.join(user_dir, "ISCO")
    
    # Select the appropriate directory based on classification type
    classification_dir = isic_dir if selected_classification == "ISIC" else isco_dir

    # Step 2: Check if user directories exist, if not, create them
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        os.makedirs(isic_dir)
        os.makedirs(isco_dir)
        st.write(f"Folder created for {user_name} with ISIC and ISCO subfolders.")
    else:
        # Ensure both subdirectories exist
        os.makedirs(isic_dir, exist_ok=True)
        os.makedirs(isco_dir, exist_ok=True)
        st.write(f"Welcome back, {user_name}. Using existing folder with {selected_classification} classification.")

    # Initialize Qdrant, encoder, and fine-tuned classifier
    qdrant, encoder, fine_tuned_classifier = initialize_models()

    # Model selection for ISIC only (since fine-tuned model is ISIC-specific)
    model_selection = "Embedding Model"
    if selected_classification == "ISIC":
        st.markdown("### 🤖 Select Model Type")
        if fine_tuned_classifier:
            model_selection = st.radio(
                "Choose the model to use for ISIC classification:",
                ("Embedding Model (Fast, Good Accuracy)", "Fine-tuned Model (Best Accuracy)"),
                help="Embedding model uses sentence similarity, Fine-tuned model uses trained DistilBERT"
            )
            
            # Show model information
            if model_selection == "Fine-tuned Model (Best Accuracy)":
                model_info = fine_tuned_classifier.get_model_info()
                if model_info['status'] == 'Loaded':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Accuracy", f"{model_info['evaluation'].get('accuracy', 'N/A'):.1%}" if isinstance(model_info['evaluation'].get('accuracy'), (int, float)) else "N/A")
                    with col2:
                        st.metric("F1 Score", f"{model_info['evaluation'].get('f1_score', 'N/A'):.1%}" if isinstance(model_info['evaluation'].get('f1_score'), (int, float)) else "N/A")
                    with col3:
                        st.metric("Training Examples", f"{model_info['evaluation'].get('training_examples', 'N/A'):,}" if isinstance(model_info['evaluation'].get('training_examples'), (int, float)) else "N/A")
        else:
            st.info("📊 Using embedding model (fine-tuned model not available)")

    # Load and encode classification data (ISIC or ISCO)
    try:
        classification_data = load_and_encode_classification_data(selected_classification, qdrant, encoder)
        if classification_data is None:
            st.stop()  # Stop execution if data loading failed
    except Exception as e:
        st.error(f"Error loading and encoding {selected_classification} data: {e}")
        st.text(traceback.format_exc())
        st.stop()

    # Initialize session state for better user experience
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    # Show processing history if any files have been processed
    if st.session_state.processed_files:
        with st.expander(f"📋 **Processing History ({len(st.session_state.processed_files)} files processed)**"):
            for idx, file_info in enumerate(st.session_state.processed_files, 1):
                st.write(f"{idx}. **{file_info['name']}** - {file_info['records']} records in {file_info['time']:.2f}s - {file_info['speed']:.1f} items/sec")
    
    # Test accuracy section
    if selected_classification == "ISIC" and fine_tuned_classifier:
        with st.expander("📈 Test Model Accuracy"):
            test_file = st.file_uploader(
                "Upload test file (Excel/CSV) with INDUSTRY and expected ISIC codes:", 
                type=['xlsx', 'csv'],
                key="accuracy_test_file"
            )
            
            if test_file:
                if st.button("Run Accuracy Test"):
                    run_accuracy_test(test_file, encoder, qdrant, fine_tuned_classifier, selected_classification)
    
    # File uploader with multiple file support
    st.markdown("### 📂 Upload Files for Processing")
    
    # Processing mode selection
    processing_mode = st.radio(
        "Choose processing mode:",
        ("Single File", "Multiple Files (Batch Processing)"),
        help="Single file for immediate processing, Multiple files for batch processing with smart optimization"
    )
    
    if processing_mode == "Single File":
        uploaded_files = st.file_uploader(
            "Choose an Excel or CSV file", 
            type=['xlsx', 'csv'], 
            accept_multiple_files=False,
            key=f"single_file_uploader_{len(st.session_state.processed_files)}"
        )
        if uploaded_files:
            uploaded_files = [uploaded_files]  # Convert to list for consistent handling
    else:
        uploaded_files = st.file_uploader(
            "Choose multiple Excel or CSV files", 
            type=['xlsx', 'csv'], 
            accept_multiple_files=True,
            key=f"multi_file_uploader_{len(st.session_state.processed_files)}",
            help="Select multiple files for batch processing. Files will be processed using smart batching based on size."
        )
    
    # Process files when uploaded
    if uploaded_files:
        # Create compact progress interface
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.markdown("---")
            
            if len(uploaded_files) == 1:
                # Single file processing
                uploaded_file = uploaded_files[0]
                st.markdown(f"### ⚡ Processing: `{uploaded_file.name}`")
                
                with st.container():
                    try:
                        # Process single file with model selection
                        use_fine_tuned = (selected_classification == "ISIC" and 
                                        model_selection == "Fine-tuned Model (Best Accuracy)" and 
                                        fine_tuned_classifier is not None)
                        
                        if use_fine_tuned:
                            result = process_file_with_fine_tuned(uploaded_file, fine_tuned_classifier, classification_dir, selected_classification)
                        else:
                            result = process_file_compact(uploaded_file, encoder, qdrant, classification_dir, selected_classification)
                        
                        # Add to processing history
                        st.session_state.processed_files.append({
                            'name': uploaded_file.name,
                            'records': result['records'],
                            'time': result['time'],
                            'speed': result['speed']
                        })
                        
                        # Show completion and download options
                        processed_filename = result['filename']
                        st.success(f"✅ **{uploaded_file.name}** processed successfully!")
                        st.info(f"💾 **Saved to server:** `{processed_filename}`")
                        
                        # Metrics display
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.metric("Records", f"{result['records']:,}")
                        with col2:
                            st.metric("Time", f"{result['time']:.1f}s")
                        with col3:
                            st.metric("Speed", f"{result['speed']:.1f}/sec")
                        
                        # Download section
                        st.markdown("---")
                        st.markdown("### 📥 Download Your Results")
                        
                        download_col, continue_col = st.columns([2, 1])
                        
                        with download_col:
                            if result['file_data'] and result['mime_type']:
                                st.download_button(
                                    label=f"📥 Download {processed_filename}",
                                    data=result['file_data'],
                                    file_name=processed_filename,
                                    mime=result['mime_type'],
                                    use_container_width=True
                                )
                        
                        with continue_col:
                            if st.button("🔄 Process More Files", use_container_width=True):
                                progress_placeholder.empty()
                                st.rerun()
                        
                        st.info("💡 **Options:**\n- Click **Download** to get the file directly\n- Click **Process More Files** to continue with additional files\n- All files are saved to your server folder")
                        
                    except Exception as e:
                        st.error(f"❌ **Error processing {uploaded_file.name}:** {e}")
                        with st.expander("🔍 Error Details"):
                            st.text(traceback.format_exc())
                        
                        if st.button("🔄 Try Again"):
                            st.rerun()
            
            else:
                # Multiple file processing
                st.markdown(f"### 🚀 Batch Processing: {len(uploaded_files)} files")
                
                # Initialize progress tracker
                progress_tracker = MultiFileProgressTracker(len(uploaded_files))
                
                # Show file analysis
                st.write("**📋 Files to process:**")
                for i, file in enumerate(uploaded_files, 1):
                    size_mb = file.size / (1024 * 1024)
                    st.write(f"{i}. `{file.name}` ({size_mb:.1f} MB)")
                
                # Start processing
                start_time = time.time()
                
                try:
                    # Process multiple files with hybrid approach
                    use_fine_tuned = (selected_classification == "ISIC" and 
                                    model_selection == "Fine-tuned Model (Best Accuracy)" and 
                                    fine_tuned_classifier is not None)
                    
                    all_results = process_multiple_files_hybrid(
                        uploaded_files, encoder, qdrant, classification_dir, 
                        selected_classification, progress_tracker, fine_tuned_classifier, use_fine_tuned
                    )
                    
                    total_time = time.time() - start_time
                    successful_results = [r for r in all_results if 'error' not in r]
                    failed_results = [r for r in all_results if 'error' in r]
                    
                    # Update session history
                    for result in successful_results:
                        st.session_state.processed_files.append({
                            'name': result.get('filename', 'Unknown'),
                            'records': result.get('records', 0),
                            'time': result.get('time', 0),
                            'speed': result.get('speed', 0)
                        })
                    
                    # Show batch completion summary
                    st.markdown("---")
                    st.success(f"🎉 **Batch Processing Complete!**")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Files", len(uploaded_files))
                    with col2:
                        st.metric("Successful", len(successful_results))
                    with col3:
                        st.metric("Failed", len(failed_results))
                    with col4:
                        st.metric("Total Time", f"{total_time:.1f}s")
                    
                    # Show individual file results
                    if successful_results:
                        st.markdown("### 📊 Processing Results")
                        for i, result in enumerate(successful_results, 1):
                            with st.expander(f"✅ {result['filename']} - {result['records']:,} records"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Processing Time:** {result['time']:.1f}s")
                                    st.write(f"**Speed:** {result['speed']:.1f} records/sec")
                                with col2:
                                    if result['file_data'] and result['mime_type']:
                                        st.download_button(
                                            label=f"📥 Download {result['filename']}",
                                            data=result['file_data'],
                                            file_name=result['filename'],
                                            mime=result['mime_type'],
                                            key=f"download_{i}"
                                        )
                    
                    # Bulk download option
                    if len(successful_results) > 1:
                        st.markdown("---")
                        st.markdown("### 📦 Bulk Download")
                        
                        bulk_col1, bulk_col2 = st.columns([2, 1])
                        
                        with bulk_col1:
                            zip_data = create_bulk_download_zip(successful_results, selected_classification)
                            st.download_button(
                                label=f"📦 Download All Files ({len(successful_results)} files)",
                                data=zip_data,
                                file_name=f"{selected_classification}_batch_results_{int(time.time())}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                        
                        with bulk_col2:
                            if st.button("🔄 Process More Batches", use_container_width=True):
                                progress_placeholder.empty()
                                st.rerun()
                    
                    # Show failed files if any
                    if failed_results:
                        st.markdown("### ❌ Failed Files")
                        for result in failed_results:
                            st.error(f"**{result['filename']}**: {result['error']}")
                    
                except Exception as e:
                    st.error(f"❌ **Batch processing failed:** {e}")
                    with st.expander("🔍 Error Details"):
                        st.text(traceback.format_exc())
                    
                    if st.button("🔄 Try Batch Again"):
                        st.rerun()
    
    # Show usage instructions
    if not uploaded_files and not st.session_state.processed_files:
        st.markdown("### 📖 Instructions")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 Single File Mode")
            st.info(
                "• **Upload** one Excel or CSV file\n"
                "• **Immediate processing** with real-time progress\n"
                "• **Direct download** when complete\n"
                "• Perfect for **quick processing**"
            )
        
        with col2:
            st.markdown("#### 📦 Batch Mode")
            st.info(
                "• **Upload multiple** Excel or CSV files\n"
                "• **Smart batching** based on file size\n"
                "• **Concurrent processing** for efficiency\n"
                "• **Bulk download** as ZIP file\n"
                "• Ideal for **large batches**"
            )
        
        # Show sample data info
        with st.expander("📄 Sample Data Format"):
            st.code(
                "INDUSTRY,DESCRIPTION\n"
                "Agriculture and Farming,Growing crops and livestock\n"
                "Manufacturing Electronics,Producing electronic components\n"
                "Software Development,Creating computer programs"
            )
    
    # Log processing activity
    if st.session_state.processed_files:
        log_file_path = os.path.join(outputs_dir, "processing_log.txt")
        with open(log_file_path, "a") as log_file:
            for file_info in st.session_state.processed_files:
                if f"{user_name} processed {file_info['name']}" not in open(log_file_path, 'r').read():
                    log_file.write(f"{user_name} processed {file_info['name']} - {file_info['records']} records\n")
