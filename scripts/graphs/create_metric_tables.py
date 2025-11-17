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

    # Rename architectures
    architecture_name_mapping = {
        'AnisotropicUNet3D-d2-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-D2', 
        'AnisotropicUNet3D-d3-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-D3',
        'AnisotropicUNet3D-d4-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-D4',
        'AnisotropicUNet3D-d5-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-D5',
        'AnisotropicUNet3D-d6-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-D6',
        'Anisotropicunet3D-D2-Hk(3-3-3)-Dk(1-2-2)': 'AnisoUNet-D2', 
        'Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)': 'AnisoUNet-D3',
        'Anisotropicunet3D-D4-Hk(3-3-3)-Dk(1-2-2)': 'AnisoUNet-D4',
        'Anisotropicunet3D-D5-Hk(3-3-3)-Dk(1-2-2)': 'AnisoUNet-D5',
        'Anisotropicunet3D-D6-Hk(3-3-3)Dk(1-2-2)': 'AnisoUNet-D6',
        'UNet3d-BasicUNet': 'BasicUNet3D',
        'AnisotropicUNet3DSE-d2-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-SE',
        'AnisotropicUNet3DCSAM-d2-hk(3-3-3)-dk(1-2-2)': 'AnisoUNet-CSAM',
        'AnisotropicUSENet-d2-hk(3-3-3)-dk(1-2-2)': 'AnisoUSENet',
    }
    df['Architecture'] = df['Architecture'].replace(architecture_name_mapping)
    print(f"Found architectures: {sorted(df['Architecture'].unique())}")

    # Define key metrics
    metrics = {
        'Eval_Dice_Mean': 'Evaluation Dice Mean',
        'Eval_Jaccard_Mean': 'Evaluation Jaccard Mean',
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
    
    # Average for each Architecture-Overlap combination
    grouped = df.groupby(['Architecture', 'Overlap'])
    
    for metric_col, metric_name in available_metrics.items():
        print(f"\nCreating table for: {metric_name}")
        
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
    im = plt.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto', vmin=0.2, vmax=0.7)
    
    # Set labels
    plt.xticks(range(len(pivot_table.columns)), [f"{col}px" for col in pivot_table.columns])
    plt.yticks(range(len(pivot_table.index)), 
               [arch[:30] + "..." if len(arch) > 30 else arch for arch in pivot_table.index])
    
    plt.colorbar(im, label='Score')
    
    # Find global maximum (single best cell) across the whole pivot table
    stacked = pivot_table.stack()
    best_pos = None
    if not stacked.empty:
        best_idx = stacked.idxmax()  # returns (Architecture, Overlap)
        best_val = stacked.max()
        best_i = pivot_table.index.get_loc(best_idx[0])
        best_j = pivot_table.columns.get_loc(best_idx[1])
        best_pos = (best_i, best_j)
    
    # Add annotations and highlight only the global maximum
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.iloc[i, j]
            if pd.isna(val):
                continue
            txt = f'{val:.3f}'
            if best_pos is not None and (i, j) == best_pos:
                plt.text(j, i, txt, ha='center', va='center', fontweight='bold', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            else:
                plt.text(j, i, txt, ha='center', va='center', fontsize=8)
    
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