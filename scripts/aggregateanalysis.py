import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os

def aggregate_csv_files(directory_path, output_path):
    """
    Aggregate multiple CSV files from a directory into a summary with means and standard deviations as separate columns.
    
    Args:
        directory_path: Path to directory containing CSV files
        output_path: Path to save the aggregated CSV file
    """
    
    # Convert to Path object and find all CSV files
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory not found: {directory_path}")
        return
    
    if not directory.is_dir():
        print(f"Path is not a directory: {directory_path}")
        return
    
    # Find all CSV files in the directory
    csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in directory: {directory_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process in {directory_path}:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Read and combine all CSV files
    all_data = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows from {csv_file.name}")
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {csv_file.name}: {e}")
            continue
    
    if not all_data:
        print("No valid data found in CSV files")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    # Filter rows where Train_Quad equals Test_Quad
    filtered_df = combined_df[combined_df['Train_Quad'] == combined_df['Test_Quad']].copy()
    print(f"After filtering Train_Quad == Test_Quad: {filtered_df.shape}")
    
    if len(filtered_df) == 0:
        print("No rows found where Train_Quad equals Test_Quad")
        return
    
    # Extract architecture from Model_Signature (convert to title case for consistency)
    filtered_df['Architecture'] = filtered_df['Model_Signature'].str.title()
    
    # Rename Test_Quad to Quad for consistency with plotting script
    filtered_df['Quad'] = filtered_df['Test_Quad']
    
    print(f"Found quadrants: {sorted(filtered_df['Quad'].unique())}")
    print(f"Found architectures: {sorted(filtered_df['Architecture'].unique())}")
    
    # Define columns to aggregate (all numeric columns except identifiers)
    exclude_columns = [
        'Run_Name', 'Overlap', 'Database_Path', 'Run_Name_Internal', 
        'Run_ID', 'Model_Signature', 'Train_Quad', 'Test_Quad', 'Architecture', 'Quad'
    ]
    
    numeric_columns = [col for col in filtered_df.columns 
                      if col not in exclude_columns and 
                      pd.api.types.is_numeric_dtype(filtered_df[col])]
    
    print(f"Aggregating {len(numeric_columns)} numeric columns")
    
    # Group by Quad and Architecture
    grouped = filtered_df.groupby(['Quad', 'Architecture'])
    
    # Calculate means and standard deviations separately
    means = grouped[numeric_columns].mean().reset_index()
    stds = grouped[numeric_columns].std().reset_index()
    
    print(f"Means shape: {means.shape}")
    print(f"Stds shape: {stds.shape}")
    
    # Create the result dataframe by merging means and stds
    result_df = means.copy()
    
    # Add standard deviation columns with _Std suffix
    for col in numeric_columns:
        std_col_name = f"{col}_Std"
        result_df[std_col_name] = stds[col].values
    
    # Sort by Quad and Architecture
    result_df = result_df.sort_values(['Quad', 'Architecture']).reset_index(drop=True)
    
    # Add Tile_Overlap column (assuming 0 for all, adjust if needed)
    result_df['Tile_Overlap'] = 0
    
    # Save the aggregated results
    output_path = Path(output_path)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nAggregation complete!")
    print(f"Saved aggregated results to: {output_path}")
    print(f"Final shape: {result_df.shape}")
    
    # Print summary
    print("\n" + "="*60)
    print("AGGREGATION SUMMARY")
    print("="*60)
    
    print(f"Total rows: {len(result_df)}")
    print(f"Quadrants: {sorted(result_df['Quad'].unique())}")
    print(f"Architectures: {sorted(result_df['Architecture'].unique())}")
    
    # Show sample counts per group
    print(f"\nSample counts per group:")
    sample_counts = filtered_df.groupby(['Quad', 'Architecture']).size().reset_index(name='Count')
    for _, row in sample_counts.iterrows():
        print(f"  {row['Quad']} + {row['Architecture']}: {row['Count']} samples")
    
    # Show first few rows as example
    print(f"\nFirst few columns of aggregated data:")
    display_cols = ['Quad', 'Architecture'] + numeric_columns[:3] + [f"{numeric_columns[0]}_Std"]
    available_display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[available_display_cols].head())
    
    return result_df

def main():
    """Main function to run the aggregation."""
    parser = argparse.ArgumentParser(
        description='Aggregate CSV files from a directory into summary statistics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all CSV files in current directory
  python aggregateanalysis.py . -o aggregated_results.csv
  
  # Aggregate CSV files from specific directory
  python aggregateanalysis.py /path/to/csv/files -o summary.csv
  
  # Aggregate CSV files from results directory
  python aggregateanalysis.py ./results -o quad_analysis.csv
        """
    )
    
    parser.add_argument('directory', 
                       help='Directory containing CSV files to aggregate')
    parser.add_argument('-o', '--output', default='aggregated_results.csv',
                       help='Output CSV file path (default: aggregated_results.csv)')
    
    args = parser.parse_args()
    
    try:
        result_df = aggregate_csv_files(args.directory, args.output)
        
        if result_df is not None:
            print(f"\n" + "="*60)
            print("SUCCESS")
            print("="*60)
            print(f"Aggregated data saved to: {args.output}")
            print("You can now use this file with your plotting script:")
            print(f"python quadbasedanalysis.py {args.output}")
        
    except Exception as e:
        print(f"Error during aggregation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()