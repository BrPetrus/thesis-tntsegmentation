import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import os


def aggregate_csv_files(directory_path, output_path, overlap_filter=None):
    """
    Aggregate multiple CSV files from a directory into a summary with means and standard deviations as separate columns.

    Args:
        directory_path: Path to directory containing CSV files
        output_path: Path to save the aggregated CSV file
        overlap_filter: If specified, only include rows with this overlap value
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
    filtered_df = combined_df[
        combined_df["Train_Quad"] == combined_df["Test_Quad"]
    ].copy()
    print(f"After filtering Train_Quad == Test_Quad: {filtered_df.shape}")

    # Apply overlap filter if specified
    if overlap_filter is not None:
        initial_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df["Overlap"] == overlap_filter].copy()
        print(
            f"After filtering Overlap == {overlap_filter}: {filtered_df.shape} (removed {initial_count - len(filtered_df)} rows)"
        )

        if len(filtered_df) == 0:
            print(f"No rows found with Overlap = {overlap_filter}")
            return

    if len(filtered_df) == 0:
        print("No rows found where Train_Quad equals Test_Quad")
        return

    # Extract architecture from Model_Signature (convert to title case for consistency)
    filtered_df["Architecture"] = filtered_df["Model_Signature"].str.title()

    # Rename Test_Quad to Quad for consistency with plotting script
    filtered_df["Quad"] = filtered_df["Test_Quad"]

    print(f"Found quadrants: {sorted(filtered_df['Quad'].unique())}")
    print(f"Found architectures: {sorted(filtered_df['Architecture'].unique())}")

    # Define columns to aggregate (all numeric columns except identifiers)
    exclude_columns = [
        "Run_Name",
        "Overlap",
        "Database_Path",
        "Run_Name_Internal",
        "Run_ID",
        "Model_Signature",
        "Train_Quad",
        "Test_Quad",
        "Architecture",
        "Quad",
    ]

    numeric_columns = [
        col
        for col in filtered_df.columns
        if col not in exclude_columns
        and pd.api.types.is_numeric_dtype(filtered_df[col])
    ]

    print(f"Aggregating {len(numeric_columns)} numeric columns")

    # Group by Quad and Architecture (and Overlap if not filtered)
    if overlap_filter is not None:
        # If overlap is filtered, group only by Architecture and Quad
        grouped = filtered_df.groupby(["Architecture", "Quad"])
        grouping_cols = ["Architecture", "Quad"]
    else:
        # If no overlap filter, group by Architecture, Quad, and Overlap
        grouped = filtered_df.groupby(["Architecture", "Quad", "Overlap"])
        grouping_cols = ["Architecture", "Quad", "Overlap"]

    # Calculate means and standard deviations separately
    means = grouped[numeric_columns].mean().reset_index()
    stds = grouped[numeric_columns].std().reset_index()

    print(f"Means shape: {means.shape}")
    print(f"Stds shape: {stds.shape}")

    # Create the result dataframe by merging means and stds
    result_df = means.copy()

    # Add the overlap value as a column if it was filtered
    if overlap_filter is not None:
        result_df["Overlap"] = overlap_filter

    # Add standard deviation columns with _Std suffix
    for col in numeric_columns:
        std_col_name = f"{col}_Std"
        result_df[std_col_name] = stds[col].values

    # Sort by Architecture, Quad, and Overlap
    sort_cols = ["Architecture", "Quad"]
    if "Overlap" in result_df.columns:
        sort_cols.append("Overlap")
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    # Save the aggregated results
    output_path = Path(output_path)
    result_df.to_csv(output_path, index=False)

    print(f"\nAggregation complete!")
    print(f"Saved aggregated results to: {output_path}")
    print(f"Final shape: {result_df.shape}")

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(result_df)}")
    print(f"Quadrants: {sorted(result_df['Quad'].unique())}")
    print(f"Architectures: {sorted(result_df['Architecture'].unique())}")
    if "Overlap" in result_df.columns:
        print(f"Overlap values: {sorted(result_df['Overlap'].unique())}")

    # Show sample counts per group
    if overlap_filter is not None:
        print(
            f"\nSample counts per (Architecture, Quad) with Overlap={overlap_filter}:"
        )
        sample_counts = (
            filtered_df.groupby(["Architecture", "Quad"])
            .size()
            .reset_index(name="Count")
        )
        for _, row in sample_counts.iterrows():
            print(
                f"  Arch={row['Architecture']} | Quad={row['Quad']}: {row['Count']} samples"
            )
    else:
        print(f"\nSample counts per (Architecture, Quad, Overlap):")
        sample_counts = (
            filtered_df.groupby(["Architecture", "Quad", "Overlap"])
            .size()
            .reset_index(name="Count")
        )
        for _, row in sample_counts.iterrows():
            print(
                f"  Arch={row['Architecture']} | Quad={row['Quad']} | Overlap={row['Overlap']}: {row['Count']} samples"
            )

    # Show first few columns of aggregated data
    print(f"\nFirst few columns of aggregated data:")
    display_cols = ["Architecture", "Quad"]
    if "Overlap" in result_df.columns:
        display_cols.append("Overlap")
    display_cols.extend(numeric_columns[:3] + [f"{numeric_columns[0]}_Std"])
    available_display_cols = [col for col in display_cols if col in result_df.columns]
    print(result_df[available_display_cols].head())

    return result_df


def aggregate_csv_files_across_quads(directory_path, output_path, overlap_filter=None):
    """
    Aggregate CSV files and calculate statistics across quadrants (not within quadrants).
    This gives overall architecture performance regardless of quadrant.

    Args:
        directory_path: Path to directory containing CSV files
        output_path: Path to save the aggregated CSV file
        overlap_filter: If specified, only include rows with this overlap value
    """

    # Use the existing aggregation logic but modify grouping
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
    filtered_df = combined_df[
        combined_df["Train_Quad"] == combined_df["Test_Quad"]
    ].copy()
    print(f"After filtering Train_Quad == Test_Quad: {filtered_df.shape}")

    # Apply overlap filter if specified
    if overlap_filter is not None:
        initial_count = len(filtered_df)
        filtered_df = filtered_df[filtered_df["Overlap"] == overlap_filter].copy()
        print(
            f"After filtering Overlap == {overlap_filter}: {filtered_df.shape} (removed {initial_count - len(filtered_df)} rows)"
        )

        if len(filtered_df) == 0:
            print(f"No rows found with Overlap = {overlap_filter}")
            return

    if len(filtered_df) == 0:
        print("No rows found where Train_Quad equals Test_Quad")
        return

    # Extract architecture from Model_Signature
    filtered_df["Architecture"] = filtered_df["Model_Signature"].str.title()

    print(f"Found quadrants: {sorted(filtered_df['Test_Quad'].unique())}")
    print(f"Found architectures: {sorted(filtered_df['Architecture'].unique())}")

    # Define columns to aggregate (all numeric columns except identifiers)
    exclude_columns = [
        "Run_Name",
        "Overlap",
        "Database_Path",
        "Run_Name_Internal",
        "Run_ID",
        "Model_Signature",
        "Train_Quad",
        "Test_Quad",
        "Architecture",
    ]

    numeric_columns = [
        col
        for col in filtered_df.columns
        if col not in exclude_columns
        and pd.api.types.is_numeric_dtype(filtered_df[col])
    ]

    print(f"Aggregating {len(numeric_columns)} numeric columns")

    # Group ONLY by Architecture (across all quadrants)
    if overlap_filter is not None:
        grouped = filtered_df.groupby(["Architecture"])
        grouping_cols = ["Architecture"]
    else:
        grouped = filtered_df.groupby(["Architecture", "Overlap"])
        grouping_cols = ["Architecture", "Overlap"]

    # Calculate means and standard deviations across quadrants
    means = grouped[numeric_columns].mean().reset_index()
    stds = grouped[numeric_columns].std().reset_index()

    print(f"Means shape: {means.shape}")
    print(f"Stds shape: {stds.shape}")

    # Create the result dataframe
    result_df = means.copy()

    # Add the overlap value as a column if it was filtered
    if overlap_filter is not None:
        result_df["Overlap"] = overlap_filter

    # Add standard deviation columns with _Std suffix
    for col in numeric_columns:
        std_col_name = f"{col}_Std"
        result_df[std_col_name] = stds[col].values

    # Sort by Architecture and Overlap
    sort_cols = ["Architecture"]
    if "Overlap" in result_df.columns:
        sort_cols.append("Overlap")
    result_df = result_df.sort_values(sort_cols).reset_index(drop=True)

    # Save the aggregated results
    output_path = Path(output_path)
    result_df.to_csv(output_path, index=False)

    print(f"\nAggregation complete!")
    print(f"Saved aggregated results to: {output_path}")
    print(f"Final shape: {result_df.shape}")

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-QUADRANT AGGREGATION SUMMARY")
    print("=" * 60)
    print(f"Total architecture entries: {len(result_df)}")
    print(f"Architectures: {sorted(result_df['Architecture'].unique())}")
    if "Overlap" in result_df.columns:
        print(f"Overlap values: {sorted(result_df['Overlap'].unique())}")

    # Show sample counts per group
    if overlap_filter is not None:
        print(f"\nSample counts per Architecture with Overlap={overlap_filter}:")
        sample_counts = (
            filtered_df.groupby(["Architecture"]).size().reset_index(name="Count")
        )
        for _, row in sample_counts.iterrows():
            print(
                f"  Arch={row['Architecture']}: {row['Count']} samples (across all quads)"
            )
    else:
        print(f"\nSample counts per (Architecture, Overlap):")
        sample_counts = (
            filtered_df.groupby(["Architecture", "Overlap"])
            .size()
            .reset_index(name="Count")
        )
        for _, row in sample_counts.iterrows():
            print(
                f"  Arch={row['Architecture']} | Overlap={row['Overlap']}: {row['Count']} samples (across all quads)"
            )

    return result_df


def main():
    """Main function to run the aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate CSV files from a directory into summary statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Aggregate all CSV files in current directory
  python aggregateanalysis.py . -o aggregated_results.csv
  
  # Aggregate CSV files from specific directory
  python aggregateanalysis.py /path/to/csv/files -o summary.csv
  
  # Aggregate only results with 20px overlap
  python aggregateanalysis.py ./results -o overlap20_results.csv --overlap 20
  
  # Aggregate only results with 0px overlap (no overlap)
  python aggregateanalysis.py ./results -o no_overlap_results.csv --overlap 0
  
  # Aggregate by (Architecture, Quad, Overlap) - default behavior
  python aggregateanalysis.py ./results -o quad_results.csv --overlap 20
  
  # Aggregate across quadrants by Architecture only
  python aggregateanalysis.py ./results -o arch_results.csv --overlap 20 --across-quads
  
  # Compare the two approaches:
  python aggregateanalysis.py ./results -o quad_20px.csv --overlap 20
  python aggregateanalysis.py ./results -o arch_20px.csv --overlap 20 --across-quads
        """,
    )

    parser.add_argument("directory", help="Directory containing CSV files to aggregate")
    parser.add_argument(
        "-o",
        "--output",
        default="aggregated_results.csv",
        help="Output CSV file path (default: aggregated_results.csv)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Filter results to only include this overlap value (e.g., 0, 10, 20, 30, 40)",
    )
    parser.add_argument(
        "--across-quads",
        action="store_true",
        help="Aggregate across quadrants (by Architecture only) instead of within quadrants",
    )

    args = parser.parse_args()

    try:
        if args.across_quads:
            print(
                "Using CROSS-QUADRANT aggregation (Architecture performance across all quadrants)"
            )
            result_df = aggregate_csv_files_across_quads(
                args.directory, args.output, args.overlap
            )
            analysis_script = "archcomparisonquad.py"
        else:
            print(
                "Using PER-QUADRANT aggregation (Architecture performance within each quadrant)"
            )
            result_df = aggregate_csv_files(args.directory, args.output, args.overlap)
            analysis_script = "quadbasedanalysis.py"

        if result_df is not None:
            print(f"\n" + "=" * 60)
            print("SUCCESS")
            print("=" * 60)
            print(f"Aggregated data saved to: {args.output}")
            overlap_msg = (
                f" with overlap={args.overlap}" if args.overlap is not None else ""
            )
            print(f"Results{overlap_msg} ready for analysis:")
            print(f"python {analysis_script} {args.output}")

    except Exception as e:
        print(f"Error during aggregation: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Set global font size
    plt.rcParams["font.size"] = 14

    main()
