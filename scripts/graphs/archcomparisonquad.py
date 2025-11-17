import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

def create_architecture_comparison_plots(csv_path, output_dir='./plots'):
    """Create bar plots comparing architectures across all quadrants with mean±std."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
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
        # 'Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)': 'AnisoUNet-3D',
        'Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)': 'AnisoUNet-2D',
        'Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Csam': 'AnisoUNet-CSAM-3D',
        'Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)-Csam': 'AnisoUNet-CSAM-2D',
        'Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Usenet-Sf16': 'AnisoUNet-UseNet-3D',
        'Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)-Usenet-Sf16': 'AnisoUNet-UseNet-2D'
    }
    df['Architecture'] = df['Architecture'].replace(architecture_name_mapping)
    print(f"Found architectures: {sorted(df['Architecture'].unique())}")
    print(f"Data loaded: {df.shape}")
    print(f"Architectures: {len(df['Architecture'].unique())}")
    
    # Define key Dice and Jaccard metrics to plot
    metrics = {
        'Eval_Dice_Mean': 'Evaluation Dice',
        'Eval_Jaccard_Mean': 'Evaluation Jaccard', 
        'Postprocess_Overall_Dice': 'Postprocess Overall Dice',
        'Postprocess_Overall_Jaccard': 'Postprocess Overall Jaccard',
        'Postprocess_Matched_Dice': 'Postprocessed Matched Tunnels Dice',
        'Postprocess_Matched_Jaccard': 'Postprocessed Matched Tunnels Jaccard',
        'Postprocess_Clean_Matched_Dice': 'Postprocessed Cleanly Matched Tunnels Dice',
        'Postprocess_Clean_Matched_Jaccard': 'Postprocessed Cleanly Matched Tunnels Jaccard',
        'Tunnel_Dice': 'Tunnel Dice',
        'Tunnel_Jaccard': 'Tunnel Jaccard'
    }
    
    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}
    
    if not available_metrics:
        print("No metrics found!")
        return
    
    print(f"Creating plots for {len(available_metrics)} metrics")
    
    # Get unique architectures
    architectures = sorted(df['Architecture'].unique())
    
    # For each metric, calculate mean and std
    for metric_col, metric_name in available_metrics.items():
        std_col = f"{metric_col}_Std"
        
        # Check if std column exists
        if std_col not in df.columns:
            print(f"Warning: {std_col} not found, using zero errors")
            df[std_col] = 0
        
        # Calculate overall means and stds per architecture (across all quadrants)
        arch_stats = []
        
        for arch in architectures:
            arch_data = df[df['Architecture'] == arch]
            
            if not arch_data.empty:
                overall_mean = arch_data[metric_col].item()
                assert arch_data[metric_col].size == 1
                overall_std = arch_data[std_col].item()
                arch_stats.append({
                    'Architecture': arch,
                    'Mean': overall_mean,
                    'Std': overall_std if pd.notna(overall_std) else 0
                })
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        means = [stat['Mean'] for stat in arch_stats]
        stds = [stat['Std'] for stat in arch_stats]
        arch_labels = [stat['Architecture'] for stat in arch_stats]
        
        # Shorten architecture names
        short_labels = []
        for arch in arch_labels:
            short_name = arch.replace('AnisotropicUNet3D-', '').replace('-hk(3-3-3)-dk(1-2-2)', '')
            short_labels.append(short_name)
        
        # Create bar plot
        x_pos = np.arange(len(short_labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                     color=plt.cm.Set3(np.linspace(0, 1, len(short_labels))))
        
        # Add value labels on bars
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                       f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(f'{metric_name} - Architecture Comparison (Across All Quadrants)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Architecture')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 0.8)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{metric_col.lower()}_architecture_comparison.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {plot_path}")
    
    # Create comprehensive summary
    create_comprehensive_summary(df, available_metrics, output_dir)

def create_comprehensive_summary(df, available_metrics, output_dir):
    """Create a comprehensive summary showing all metrics for all architectures."""
    
    print("\nCreating comprehensive summary...")
    
    architectures = sorted(df['Architecture'].unique())
    
    # Calculate stats for all metrics and architectures
    summary_data = []
    
    for arch in architectures:
        arch_data = df[df['Architecture'] == arch]
        arch_summary = {'Architecture': arch}
        
        for metric_col, metric_name in available_metrics.items():
            std_col = f"{metric_col}_Std"
            if not arch_data.empty:
                mean_val = arch_data[metric_col].item()
                std_val = arch_data[std_col].item()
                assert arch_data[metric_col].size == 1
                arch_summary[f'{metric_name}_Mean'] = mean_val
                arch_summary[f'{metric_name}_Std'] = std_val if pd.notna(std_val) else 0
            else:
                arch_summary[f'{metric_name}_Mean'] = 0
                arch_summary[f'{metric_name}_Std'] = 0
        
        summary_data.append(arch_summary)
    
    # Create subplot for each metric
    n_metrics = len(available_metrics)
    cols = 2
    rows = int(np.ceil(n_metrics / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5*rows))
    if rows == 1:
        axes = [axes] if n_metrics == 1 else axes
    else:
        axes = axes.flatten()
    
    fig.suptitle('Architecture Performance Summary - All Metrics', fontsize=16, fontweight='bold')
    
    for idx, (metric_col, metric_name) in enumerate(available_metrics.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        means = [data[f'{metric_name}_Mean'] for data in summary_data]
        stds = [data[f'{metric_name}_Std'] for data in summary_data]
        arch_labels = [data['Architecture'] for data in summary_data]
        
        # Shorten names
        short_labels = []
        for arch in arch_labels:
            short_name = arch.replace('AnisotropicUNet3D-', '').replace('-hk(3-3-3)-dk(1-2-2)', '')
            short_labels.append(short_name)
        
        x_pos = np.arange(len(short_labels))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7)
        
        # Highlight best performer
        best_idx = np.argmax(means)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_alpha(0.9)
        
        # Add value labels
        for j, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            if mean > 0:
                label = f'{mean:.3f}' if j != best_idx else f'{mean:.3f}\n(BEST)'
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.005,
                       label, ha='center', va='bottom', fontsize=8, 
                       fontweight='bold' if j == best_idx else 'normal')
        
        ax.set_title(metric_name, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if means and max(means) > 0:
            max_val = max([m + s for m, s in zip(means, stds) if m > 0])
            ax.set_ylim(0, max_val * 1.2)
    
    # Hide empty subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save summary plot
    summary_path = output_dir / 'architecture_performance_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary: {summary_path}")
    
    # Print rankings to console
    print("\nARCHITECTURE RANKINGS:")
    print("=" * 70)
    for metric_col, metric_name in available_metrics.items():
        print(f"\n{metric_name}:")
        print("-" * 50)
        
        # Sort architectures by this metric
        metric_rankings = []
        for data in summary_data:
            metric_rankings.append({
                'Architecture': data['Architecture'].replace('AnisotropicUNet3D-', '').replace('-hk(3-3-3)-dk(1-2-2)', ''),
                'Mean': data[f'{metric_name}_Mean'],
                'Std': data[f'{metric_name}_Std']
            })
        
        metric_rankings.sort(key=lambda x: x['Mean'], reverse=True)
        
        for i, ranking in enumerate(metric_rankings[:5]):  # Top 5
            print(f"  {i+1:2d}. {ranking['Architecture']:<15} = {ranking['Mean']:.4f} ± {ranking['Std']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Create architecture comparison plots from quad-based CSV.')
    parser.add_argument('csv_path', help='Path to quad-based aggregated CSV file')
    parser.add_argument('-o', '--output-dir', default='./plots', help='Output directory for plots')
    
    args = parser.parse_args()
    
    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}")
        return
    
    try:
        create_architecture_comparison_plots(args.csv_path, args.output_dir)
        print(f"\nDone! Check '{args.output_dir}' for plots.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()