#!/usr/bin/env python3
"""
Convert multiple CSV logs to separate TensorBoard runs.
Automatically detects all CSV files and all columns dynamically.
"""

import pandas as pd
import os
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def get_run_name_from_filename(filename):
    """Generate a meaningful run name from the filename"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Map common patterns to readable names
    if base_name.startswith("resume_log_"):
        # Extract timestamp or number
        suffix = base_name.replace("resume_log_", "")
        return f"resume_{suffix}"
    elif base_name == "training_log":
        return "initial_training"
    elif base_name.startswith("eval_"):
        return f"evaluation_{base_name.replace('eval_', '')}"
    else:
        return base_name

def convert_csv_to_tensorboard(csv_path, output_base_dir="tensorboard"):
    """
    Convert a CSV file to TensorBoard events.
    Automatically detects all columns and logs them.
    """
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return False
    
    try:
        # Read CSV - auto-detect all columns
        df = pd.read_csv(csv_path)
        
        # Get run name from filename
        run_name = get_run_name_from_filename(csv_path)
        
        # Create output directory for this run
        output_dir = os.path.join(output_base_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all columns except step/episode (these are used as x-axis)
        all_columns = list(df.columns)
        step_column = None
        
        # Auto-detect which column to use as steps
        if 'step' in all_columns:
            step_column = 'step'
        elif 'steps' in all_columns:
            step_column = 'steps'
        elif 'timestep' in all_columns:
            step_column = 'timestep'
        elif 'Episode' in all_columns:
            step_column = 'Episode'
        elif 'episode' in all_columns:
            step_column = 'episode'
        else:
            # Use index as step if no step column found
            print(f"   ⚠️ No step column found, using row index")
            df['index'] = df.index
            step_column = 'index'
        
        # Columns to use as metrics (all except the step column)
        metric_columns = [col for col in all_columns if col != step_column]
        
        print(f"📊 Converting {os.path.basename(csv_path)} → {run_name}")
        print(f"   Rows: {len(df)}")
        print(f"   Step column: {step_column}")
        print(f"   Metrics: {len(metric_columns)} columns - {', '.join(metric_columns[:5])}{'...' if len(metric_columns) > 5 else ''}")
        
        # Create TensorBoard writer
        writer = SummaryWriter(output_dir)
        
        # Log each metric column
        for col in metric_columns:
            # Skip if the column is all NaN or empty
            if df[col].isna().all():
                print(f"   ⚠️ Skipping {col} - all NaN")
                continue
            
            # Log each row's data
            for idx, row in df.iterrows():
                step_value = int(row[step_column]) if pd.notna(row[step_column]) else idx
                value = row[col]
                
                # Only log if value is not NaN
                if pd.notna(value):
                    writer.add_scalar(col, float(value), step_value)
        
        writer.close()
        print(f"   ✅ Saved to: {output_dir}\n")
        return True
        
    except Exception as e:
        print(f"❌ Error converting {csv_path}: {str(e)}")
        return False

def main():
    # Get all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("❌ No CSV files found in current directory!")
        print(f"   Current directory: {os.getcwd()}")
        return
    
    print(f"🔍 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f}")
    print()
    
    # Ask user if they want to continue
    response = input("Continue with conversion? (y/n): ").strip().lower()
    if response != 'y':
        print("❌ Conversion cancelled.")
        return
    
    # Delete old tensorboard folder if exists
    output_dir = "tensorboard"
    if os.path.exists(output_dir):
        print(f"🗑️ Removing existing {output_dir} folder...")
        shutil.rmtree(output_dir)
    
    print(f"\n🔄 Starting conversion...\n")
    
    # Convert each file
    success_count = 0
    for csv_file in csv_files:
        if convert_csv_to_tensorboard(csv_file, output_dir):
            success_count += 1
    
    print("="*60)
    print(f"✅ Conversion complete! {success_count}/{len(csv_files)} files converted successfully.")
    print(f"📁 Output directory: {output_dir}/")
    print("\n🚀 To view in TensorBoard, run:")
    print(f"   tensorboard --logdir={output_dir} --bind_all")
    print("\n💡 In TensorBoard, click the 'Run' dropdown to see all runs:")
    
    # List all runs created
    if os.path.exists(output_dir):
        runs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        for run in runs:
            print(f"   - {run}")
    print("="*60)

if __name__ == '__main__':
    main()