"""
Simple script to view and manage processing logs
Usage: python view_logs.py [--failed] [--success] [--user NAME] [--type ISIC|ISCO]
"""
import json
import os
import sys
from file_logger import get_failed_files, get_successful_files, SUCCESS_LOG_FILE, FAILED_LOG_FILE

def print_file_entry(entry, index):
    """Print a formatted file entry"""
    print(f"\n{'='*80}")
    print(f"Entry #{index}")
    print(f"{'='*80}")
    print(f"Status: {entry.get('status', 'UNKNOWN')}")
    print(f"Filename: {entry.get('filename', 'Unknown')}")
    print(f"Timestamp: {entry.get('timestamp', 'Unknown')}")
    print(f"User: {entry.get('user_name', 'Unknown')}")
    print(f"Classification: {entry.get('classification_type', 'Unknown')}")
    print(f"Model: {entry.get('model_type', 'Unknown')}")
    
    if entry.get('status') == 'SUCCESS':
        print(f"Records Processed: {entry.get('records_processed', 0):,}")
        print(f"Processing Time: {entry.get('processing_time_seconds', 0):.2f}s")
        print(f"Speed: {entry.get('speed_records_per_sec', 0):.2f} records/sec")
        print(f"Output Path: {entry.get('output_path', 'N/A')}")
    else:
        print(f"Error Type: {entry.get('error_type', 'Unknown')}")
        print(f"Error Message: {entry.get('error_message', 'Unknown error')}")
        print(f"Records Attempted: {entry.get('records_attempted', 0)}")
        print(f"Processing Time: {entry.get('processing_time_seconds', 0):.2f}s")
        print(f"Needs Retry: {entry.get('needs_retry', True)}")

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description='View processing logs')
    parser.add_argument('--failed', action='store_true', help='Show failed files')
    parser.add_argument('--success', action='store_true', help='Show successful files')
    parser.add_argument('--user', type=str, help='Filter by user name')
    parser.add_argument('--type', type=str, choices=['ISIC', 'ISCO'], help='Filter by classification type')
    parser.add_argument('--count', type=int, default=50, help='Maximum number of entries to show (default: 50)')
    
    args = parser.parse_args()
    
    # If no specific flag, show both
    show_failed = args.failed or (not args.failed and not args.success)
    show_success = args.success or (not args.failed and not args.success)
    
    count = 0
    
    if show_failed:
        print("\n" + "="*80)
        print("FAILED FILES")
        print("="*80)
        failed_files = get_failed_files(user_name=args.user, classification_type=args.type)
        
        if not failed_files:
            print("\n✅ No failed files found!")
        else:
            print(f"\nFound {len(failed_files)} failed file(s)")
            # Show most recent first
            for idx, entry in enumerate(reversed(failed_files[-args.count:]), 1):
                print_file_entry(entry, idx)
                count += 1
                if count >= args.count:
                    break
    
    if show_success:
        print("\n" + "="*80)
        print("SUCCESSFUL FILES")
        print("="*80)
        successful_files = get_successful_files(user_name=args.user, classification_type=args.type)
        
        if not successful_files:
            print("\n📝 No successful files found in logs")
        else:
            print(f"\nFound {len(successful_files)} successful file(s)")
            # Show most recent first
            for idx, entry in enumerate(reversed(successful_files[-args.count:]), 1):
                print_file_entry(entry, idx)
                count += 1
                if count >= args.count:
                    break
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Log files location: logs/")
    print(f"  - Successful: {SUCCESS_LOG_FILE}")
    print(f"  - Failed: {FAILED_LOG_FILE}")
    
    if show_failed:
        failed_count = len(get_failed_files(user_name=args.user, classification_type=args.type))
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} file(s) need to be retried")
            print("\nTo retry failed files:")
            print("1. Review the failed files list above")
            print("2. Fix any issues (missing columns, data format, etc.)")
            print("3. Re-upload the files through the application")

if __name__ == "__main__":
    main()

