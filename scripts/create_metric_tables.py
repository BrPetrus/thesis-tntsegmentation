import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def create_metric_tables(csv_path, output_dir='./tables'):
    """Create 2D tables showing Architecture vs Overlap with mean values only."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    print(f"Data shape: {df.shape}")
    print(f"Found overlaps: {sorted(df['Overlap'].unique())}")
    
    # Define key metrics
    metrics = {
        'Eval_Dice_Mean': 'Evaluation Dice Mean',
        'Eval_Jaccard_Mean': 'Evaluation Jaccard Mean',
        # 'Eval_Accuracy_Mean': 'Evaluation Accuracy Mean',
        # 'Eval_Precision_Mean': 'Evaluation Precision Mean',
        # 'Eval_Recall_Mean': 'Evaluation Recall Mean',
        'Postprocess_Overall_Dice': 'Postprocess Overall Dice',
        'Postprocess_Overall_Jaccard': 'Postprocess Overall Jaccard',
        'Tunnel_Dice': 'Tunnel Dice',
        'Tunnel_Jaccard': 'Tunnel Jaccard'
    }
    
    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not available_metrics:
        print("No metrics found")
        return
    
    print(f"Creating tables for {len(available_metrics)} metrics")
    
    # Average across quadrants for each Architecture-Overlap combination
    grouped = df.groupby(['Architecture', 'Overlap'])
    
    for metric_col, metric_name in available_metrics.items():
        print(f"\nCreating table for: {metric_name}")
        
        # Calculate mean across quadrants
        agg_data = grouped[metric_col].mean().reset_index()
        
        # Create pivot table
        pivot_table = agg_data.pivot_table(
            values=metric_col, 
            index='Architecture', 
            columns='Overlap', 
            aggfunc='first'
        )
        
        # Create heatmap
        create_heatmap(pivot_table, metric_name, output_dir, metric_col)
        
        # Save CSV
        pivot_table.to_csv(output_dir / f"{metric_col.lower()}_table.csv")

def create_heatmap(pivot_table, metric_name, output_dir, metric_col):
    """Create heatmap with mean values and highlight best architecture for each overlap."""
    
    plt.figure(figsize=(max(12, len(pivot_table.columns) * 1.5), max(8, len(pivot_table.index) * 0.6)))
    
    # Create heatmap
    im = plt.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    plt.xticks(range(len(pivot_table.columns)), [f"{col}px" for col in pivot_table.columns])
    plt.yticks(range(len(pivot_table.index)), 
               [arch[:30] + "..." if len(arch) > 30 else arch for arch in pivot_table.index])
    
    plt.colorbar(im, label='Score')
    
    # Find best architecture for each overlap (column-wise max)
    best_archs_per_overlap = {}
    for j, col in enumerate(pivot_table.columns):
        col_data = pivot_table.iloc[:, j]
        valid_data = col_data.dropna()
        if len(valid_data) > 0:
            best_archs_per_overlap[j] = valid_data.idxmax()
    
    # Add annotations and highlight max values per overlap
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            
            if not pd.isna(val):
                # Highlight if this is the best architecture for this overlap
                if j in best_archs_per_overlap and pivot_table.index[i] == best_archs_per_overlap[j]:
                    plt.text(j, i, f'{val:.3f}', ha='center', va='center', fontweight='bold', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
                else:
                    plt.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
    
    plt.title(f'{metric_name} - Architecture vs Overlap', fontsize=14, fontweight='bold')
    plt.xlabel('Overlap (pixels)', fontsize=12)
    plt.ylabel('Architecture', fontsize=12)
    plt.tight_layout()
    
    # Save
    plot_path = output_dir / f"{metric_col.lower()}_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Create 2D metric tables from aggregated CSV.')
    parser.add_argument('csv_path', help='Path to aggregated CSV file')
    parser.add_argument('-o', '--output-dir', default='./tables', help='Output directory')
    
    args = parser.parse_args()
    
    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}")
        return
    
    try:
        create_metric_tables(args.csv_path, args.output_dir)
        print(f"\nDone. Check '{args.output_dir}' for heatmaps and CSV files.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()