# File Processing Logging Guide

This application automatically logs all file processing attempts to help you track successful and failed files.

## 📁 Log Files Location

All logs are stored in the `logs/` directory:
- **Successful files**: `logs/successful_files.log`
- **Failed files**: `logs/failed_files.log`

## 📝 Log Format

Logs are stored in JSON format (one entry per line) with the following information:

### Successful Files Log
```json
{
  "timestamp": "2025-01-20T10:30:45.123456",
  "status": "SUCCESS",
  "filename": "my_file.xlsx",
  "user_name": "John Doe",
  "classification_type": "ISIC",
  "model_type": "fine_tuned",
  "level_2_only": false,
  "records_processed": 5000,
  "processing_time_seconds": 125.5,
  "speed_records_per_sec": 39.84,
  "output_path": "outputs/John Doe/ISIC/my_file_fine_tuned.xlsx"
}
```

### Failed Files Log
```json
{
  "timestamp": "2025-01-20T10:35:12.789012",
  "status": "FAILED",
  "filename": "problematic_file.xlsx",
  "user_name": "John Doe",
  "classification_type": "ISIC",
  "model_type": "embedding",
  "error_message": "Missing 'INDUSTRY' column",
  "error_type": "ValueError",
  "records_attempted": 0,
  "processing_time_seconds": 0.5,
  "needs_retry": true,
  "traceback": "..."
}
```

## 🔍 Viewing Logs

### Method 1: Using the Application UI

1. Check the sidebar in the Streamlit app
2. Look for the **"📋 Processing Logs"** section
3. See a summary of failed files that need retry
4. Click to expand and view details

### Method 2: Using the Command Line Script

Use the `view_logs.py` script to view and filter logs:

```bash
# View all failed files
python view_logs.py --failed

# View all successful files
python view_logs.py --success

# View both (default)
python view_logs.py

# Filter by user name
python view_logs.py --failed --user "John Doe"

# Filter by classification type
python view_logs.py --failed --type ISIC

# Limit number of entries shown
python view_logs.py --failed --count 10
```

### Method 3: Direct File Access

You can open the log files directly:
- They are plain text files in JSON format
- One JSON object per line
- Easy to parse programmatically

## 🔄 Retrying Failed Files

### Step 1: Identify Failed Files

Check the failed files log or use the command:
```bash
python view_logs.py --failed
```

### Step 2: Review Error Messages

Common errors and fixes:

1. **Missing Required Column**
   - Error: "Missing 'INDUSTRY' column"
   - Fix: Ensure your file has the required column (INDUSTRY for ISIC, OCCUPATION/INDUSTRY for ISCO)

2. **File Format Issues**
   - Error: "Could not read Excel file"
   - Fix: Ensure file is valid Excel (.xlsx) or CSV format

3. **Data Quality Issues**
   - Error: "Empty or invalid data"
   - Fix: Check that INDUSTRY column has valid text entries

### Step 3: Fix and Re-upload

1. Fix the issues identified
2. Re-upload the file through the application
3. The new attempt will be logged separately

## 📊 Log Statistics

### Count Failed Files
```bash
python -c "from file_logger import get_failed_files; print(len(get_failed_files()))"
```

### Count Successful Files
```bash
python -c "from file_logger import get_successful_files; print(len(get_successful_files()))"
```

### Filter and Analyze
You can write custom Python scripts to analyze the logs:

```python
from file_logger import get_failed_files, get_successful_files
import json

# Get all failed files for a specific user
failed = get_failed_files(user_name="John Doe")

# Analyze error types
error_types = {}
for entry in failed:
    error_type = entry.get('error_type', 'Unknown')
    error_types[error_type] = error_types.get(error_type, 0) + 1

print("Error distribution:")
for error_type, count in error_types.items():
    print(f"  {error_type}: {count}")
```

## 🧹 Managing Logs

### Clear Logs (Use with Caution)
```python
from file_logger import clear_logs
clear_logs()  # This will delete all logs!
```

### Archive Old Logs
You can manually move log files to archive them:
```bash
mkdir logs/archive
mv logs/successful_files.log logs/archive/successful_2025-01-20.log
mv logs/failed_files.log logs/archive/failed_2025-01-20.log
```

## 💡 Tips

1. **Regular Review**: Check failed files log regularly to catch issues early
2. **Batch Analysis**: Use the view_logs.py script to analyze patterns in failures
3. **Backup Logs**: Consider backing up logs periodically for audit purposes
4. **Filter by Date**: Logs include timestamps, so you can filter by date ranges programmatically

## 🔧 Troubleshooting

### Logs directory doesn't exist
- It will be created automatically on first use
- If you have permission issues, create it manually: `mkdir logs`

### Can't read log files
- Ensure you have read permissions
- Check that files exist: `ls -la logs/`

### Logs are too large
- Consider archiving old logs periodically
- The logs are append-only, so they will grow over time

