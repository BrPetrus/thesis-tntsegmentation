import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_architecture_plots(csv_path, output_dir='./plots'):
    """Create bar plots for architectures with mean±std for Dice and Jaccard metrics."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    print(f"Data loaded: {df.shape}")
    print(f"Architectures: {len(df['Architecture'].unique())}")
    print(f"Overlaps: {sorted(df['Overlap'].unique())}")
    
    # Define key Dice and Jaccard metrics to plot
    metrics = {
        'Eval_Dice_Mean': 'Evaluation Dice',
        'Eval_Jaccard_Mean': 'Evaluation Jaccard', 
        'Postprocess_Overall_Dice': 'Postprocess Overall Dice',
        'Postprocess_Overall_Jaccard': 'Postprocess Overall Jaccard',
        'Tunnel_Dice': 'Tunnel Dice',
        'Tunnel_Jaccard': 'Tunnel Jaccard'
    }
    
    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not available_metrics:
        print("No metrics found!")
        return
    
    print(f"Creating plots for {len(available_metrics)} metrics")
    
    # Get unique architectures and overlaps
    architectures = sorted(df['Architecture'].unique())
    overlaps = sorted(df['Overlap'].unique())
    
    # Create plots for each metric
    for metric_col, metric_name in available_metrics.items():
        std_col = f"{metric_col}_Std"
        
        # Check if std column exists
        if std_col not in df.columns:
            print(f"Warning: {std_col} not found, using zero errors")
            df[std_col] = 0
        
        # Create subplot for each overlap
        n_overlaps = len(overlaps)
        cols = min(3, n_overlaps)
        rows = int(np.ceil(n_overlaps / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_overlaps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_overlaps > 1 else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(f'{metric_name} by Architecture and Overlap', fontsize=16, fontweight='bold')
        
        for i, overlap in enumerate(overlaps):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get data for this overlap
            overlap_data = df[df['Overlap'] == overlap]
            
            means = []
            stds = []
            arch_labels = []
            
            for arch in architectures:
                arch_data = overlap_data[overlap_data['Architecture'] == arch]
                if not arch_data.empty:
                    mean_val = arch_data[metric_col].iloc[0]
                    std_val = arch_data[std_col].iloc[0]
                    means.append(mean_val)
                    stds.append(std_val)
                    # Shorten architecture names
                    short_name = arch.replace('AnisotropicUNet3D-', '').replace('-hk(3-3-3)-dk(1-2-2)', '')
                    arch_labels.append(short_name)
                else:
                    means.append(0)
                    stds.append(0)
                    arch_labels.append(arch[:10] + "...")
            
            # Create bar plot
            x_pos = np.arange(len(arch_labels))
            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=plt.cm.Set3(np.linspace(0, 1, len(arch_labels))))
            
            # Add value labels on bars
            for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                           f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{overlap}px Overlap', fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Set y-axis limits
            if means and max(means) > 0:
                max_val = max([m + s for m, s in zip(means, stds) if m > 0])
                ax.set_ylim(0, max_val * 1.1)
        
        # Hide empty subplots
        for i in range(len(overlaps), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{metric_col.lower()}_by_architecture.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {plot_path}")
    
    # Create a summary plot with best architectures per metric
    create_summary_plot(df, available_metrics, output_dir)

def create_summary_plot(df, available_metrics, output_dir):
    """Create a summary plot showing best architecture for each metric across all overlaps."""
    
    print("\nCreating summary plot...")
    
    # For each metric, find the best architecture-overlap combination
    best_results = []
    
    for metric_col, metric_name in available_metrics.items():
        best_idx = df[metric_col].idxmax()
        best_row = df.iloc[best_idx]
        
        best_results.append({
            'Metric': metric_name,
            'Architecture': best_row['Architecture'].replace('AnisotropicUNet3D-', '').replace('-hk(3-3-3)-dk(1-2-2)', ''),
            'Overlap': best_row['Overlap'],
            'Score': best_row[metric_col],
            'Std': best_row.get(f"{metric_col}_Std", 0)
        })
    
    # Create summary bar plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    metrics_names = [r['Metric'] for r in best_results]
    scores = [r['Score'] for r in best_results]
    stds = [r['Std'] for r in best_results]
    
    x_pos = np.arange(len(metrics_names))
    bars = ax.bar(x_pos, scores, yerr=stds, capsize=5, alpha=0.7,
                 color=plt.cm.Set1(np.linspace(0, 1, len(metrics_names))))
    
    # Add labels with architecture and overlap info
    for i, (bar, result) in enumerate(zip(bars, best_results)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + result['Std'] + 0.01,
               f"{result['Architecture']}\n{result['Overlap']}px\n{result['Score']:.3f}",
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_title('Best Architecture-Overlap Combination per Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_dir / 'best_architectures_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary: {summary_path}")
    
    # Print summary to console
    print("\nBEST COMBINATIONS:")
    print("-" * 50)
    for result in best_results:
        print(f"{result['Metric']:<25}: {result['Architecture']:<15} @ {result['Overlap']:>2}px = {result['Score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Create architecture comparison plots from cross-quad CSV.')
    parser.add_argument('csv_path', help='Path to cross-quad aggregated CSV file')
    parser.add_argument('-o', '--output-dir', default='./plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}")
        return
    
    try:
        create_architecture_plots(args.csv_path, args.output_dir)
        print(f"\nDone! Check '{args.output_dir}' for plots.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()