"""
File processing logger module
Logs successful and failed file processing operations to separate log files
"""
import os
import json
from datetime import datetime
from typing import Dict, Optional

# Log directory
LOGS_DIR = "logs"
SUCCESS_LOG_FILE = os.path.join(LOGS_DIR, "successful_files.log")
FAILED_LOG_FILE = os.path.join(LOGS_DIR, "failed_files.log")

def ensure_logs_directory():
    """Create logs directory if it doesn't exist"""
    os.makedirs(LOGS_DIR, exist_ok=True)

def log_successful_file(
    filename: str,
    user_name: str,
    classification_type: str,
    model_type: str,
    records_processed: int,
    processing_time: float,
    speed: float,
    output_path: Optional[str] = None,
    level_2_only: bool = False,
    additional_info: Optional[Dict] = None
):
    """
    Log a successfully processed file
    
    Args:
        filename: Name of the processed file
        user_name: Name of the user who processed the file
        classification_type: ISIC or ISCO
        model_type: 'fine_tuned' or 'embedding'
        records_processed: Number of records successfully processed
        processing_time: Time taken to process (seconds)
        speed: Processing speed (records/second)
        output_path: Path where the processed file was saved
        level_2_only: Whether Level 2 only mode was used
        additional_info: Any additional information to log
    """
    ensure_logs_directory()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'SUCCESS',
        'filename': filename,
        'user_name': user_name,
        'classification_type': classification_type,
        'model_type': model_type,
        'level_2_only': level_2_only,
        'records_processed': records_processed,
        'processing_time_seconds': round(processing_time, 2),
        'speed_records_per_sec': round(speed, 2),
        'output_path': output_path,
    }
    
    if additional_info:
        log_entry.update(additional_info)
    
    # Append to success log file
    with open(SUCCESS_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def log_failed_file(
    filename: str,
    user_name: str,
    classification_type: str,
    model_type: str,
    error_message: str,
    error_type: Optional[str] = None,
    records_attempted: int = 0,
    processing_time: float = 0.0,
    additional_info: Optional[Dict] = None
):
    """
    Log a failed file processing attempt
    
    Args:
        filename: Name of the file that failed
        user_name: Name of the user who attempted processing
        classification_type: ISIC or ISCO
        model_type: 'fine_tuned' or 'embedding'
        error_message: Error message or description
        error_type: Type of error (e.g., 'ValueError', 'FileNotFoundError')
        records_attempted: Number of records attempted before failure
        processing_time: Time spent before failure (seconds)
        additional_info: Any additional information to log
    """
    ensure_logs_directory()
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'status': 'FAILED',
        'filename': filename,
        'user_name': user_name,
        'classification_type': classification_type,
        'model_type': model_type,
        'error_message': str(error_message),
        'error_type': error_type,
        'records_attempted': records_attempted,
        'processing_time_seconds': round(processing_time, 2),
        'needs_retry': True,
    }
    
    if additional_info:
        log_entry.update(additional_info)
    
    # Append to failed log file
    with open(FAILED_LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

def get_failed_files(user_name: Optional[str] = None, classification_type: Optional[str] = None) -> list:
    """
    Get list of failed files from the log
    
    Args:
        user_name: Filter by user name (optional)
        classification_type: Filter by classification type (optional)
    
    Returns:
        List of failed file entries
    """
    if not os.path.exists(FAILED_LOG_FILE):
        return []
    
    failed_files = []
    with open(FAILED_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    # Apply filters if provided
                    if user_name and entry.get('user_name') != user_name:
                        continue
                    if classification_type and entry.get('classification_type') != classification_type:
                        continue
                    failed_files.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return failed_files

def get_successful_files(user_name: Optional[str] = None, classification_type: Optional[str] = None) -> list:
    """
    Get list of successful files from the log
    
    Args:
        user_name: Filter by user name (optional)
        classification_type: Filter by classification type (optional)
    
    Returns:
        List of successful file entries
    """
    if not os.path.exists(SUCCESS_LOG_FILE):
        return []
    
    successful_files = []
    with open(SUCCESS_LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    # Apply filters if provided
                    if user_name and entry.get('user_name') != user_name:
                        continue
                    if classification_type and entry.get('classification_type') != classification_type:
                        continue
                    successful_files.append(entry)
                except json.JSONDecodeError:
                    continue
    
    return successful_files

def clear_logs():
    """Clear all log files (use with caution)"""
    ensure_logs_directory()
    if os.path.exists(SUCCESS_LOG_FILE):
        open(SUCCESS_LOG_FILE, 'w').close()
    if os.path.exists(FAILED_LOG_FILE):
        open(FAILED_LOG_FILE, 'w').close()

