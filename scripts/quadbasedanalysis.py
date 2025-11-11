import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_performance_plots(csv_path, output_dir='./plots'):
    """
    Create bar plots for Dice and Jaccard metrics by quadrant and architecture.
    
    Args:
        csv_path: Path to the CSV file
        output_dir: Directory to save the plots
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Separate mean and std data
    mean_df = df[~df['Quad'].str.contains('std', na=False)].copy()
    std_df = df[df['Quad'].str.contains('std', na=False)].copy()
    
    # Clean the std dataframe - remove 'std' suffix from Quad column
    std_df['Quad'] = std_df['Quad'].str.replace('std', '')
    
    print(f"Mean data shape: {mean_df.shape}")
    print(f"Std data shape: {std_df.shape}")
    
    # # Define the metrics we want to plot (Dice and Jaccard related columns)
    # dice_jaccard_columns = [
    #     'Train_Dice', 'Train_Jaccard',
    #     'Eval_Dice_Mean', 'Eval_Jaccard_Mean',
    #     'Postprocess_Overall_Dice', 'Postprocess_Overall_Jaccard',
    #     'Postprocess_Matched_Dice', 'Postprocess_Matched_Jaccard',
    #     'Postprocess_Clean_Matched_Dice', 'Postprocess_Clean_Matched_Jaccard',
    #     'Tunnel_Dice', 'Tunnel_Jaccard'
    # ]
    
    # Get unique quadrants and architectures from mean data
    quadrants = sorted(mean_df['Quad'].unique())
    architectures = sorted(mean_df['Architecture'].unique())
    
    print(f"Found quadrants: {quadrants}")
    print(f"Found architectures: {architectures}")
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create plots for each metric category
    metric_categories = {
        'Training': ['Train_Dice', 'Train_Jaccard'],
        'Evaluation': ['Eval_Dice_Mean', 'Eval_Jaccard_Mean'],
        'Postprocess_Overall': ['Postprocess_Overall_Dice', 'Postprocess_Overall_Jaccard'],
        'Postprocess_Matched': ['Postprocess_Matched_Dice', 'Postprocess_Matched_Jaccard'],
        'Postprocess_Clean': ['Postprocess_Clean_Matched_Dice', 'Postprocess_Clean_Matched_Jaccard'],
        'Tunnel': ['Tunnel_Dice', 'Tunnel_Jaccard']
    }
    
    for category_name, metrics in metric_categories.items():
        # Check if metrics exist in dataframe
        available_metrics = [m for m in metrics if m in mean_df.columns]
        if not available_metrics:
            print(f"Skipping {category_name}: metrics not found in data")
            continue
            
        print(f"\nCreating plots for {category_name}...")
        
        # Create subplots for each quadrant
        n_quads = len(quadrants)
        if n_quads <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            # If more than 4 quadrants, adjust subplot layout
            rows = int(np.ceil(n_quads / 3))
            fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
            axes = axes.flatten() if rows > 1 else [axes] if rows == 1 and n_quads == 1 else axes
        
        fig.suptitle(f'{category_name} Metrics by Architecture and Quadrant', fontsize=16, fontweight='bold')
        
        for idx, quad in enumerate(quadrants):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Filter data for current quadrant
            quad_mean_data = mean_df[mean_df['Quad'] == quad]
            quad_std_data = std_df[std_df['Quad'] == quad] if not std_df.empty else pd.DataFrame()
            
            if quad_mean_data.empty:
                ax.text(0.5, 0.5, f'No data for {quad}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{quad.upper()}')
                continue
            
            # Prepare data for plotting
            x_pos = np.arange(len(architectures))
            width = 0.35 if len(available_metrics) == 2 else 0.8 / len(available_metrics)
            
            # Create bars for each metric
            for i, metric in enumerate(available_metrics):
                values = []
                errors = []
                
                for arch in architectures:
                    # Get mean value
                    arch_mean_data = quad_mean_data[quad_mean_data['Architecture'] == arch]
                    
                    if not arch_mean_data.empty:
                        value = arch_mean_data[metric].iloc[0]
                        values.append(value if pd.notna(value) else 0)
                        
                        # Get std value if available
                        arch_std_data = quad_std_data[quad_std_data['Architecture'] == arch] if not quad_std_data.empty else pd.DataFrame()
                        if not arch_std_data.empty and metric in arch_std_data.columns:
                            std_value = arch_std_data[metric].iloc[0]
                            errors.append(std_value if pd.notna(std_value) else 0)
                        else:
                            errors.append(0)
                    else:
                        values.append(0)
                        errors.append(0)
                
                # Create bars
                if len(available_metrics) == 2:
                    offset = (i - 0.5) * width
                else:
                    offset = (i - (len(available_metrics) - 1) / 2) * width
                
                bars = ax.bar(x_pos + offset, values, width, 
                             label=metric.replace('_', ' '), 
                             alpha=0.8,
                             yerr=errors if any(errors) else None,
                             capsize=3)
                
                # Add value labels on bars
                for bar, value, error in zip(bars, values, errors):
                    if value > 0:
                        height = bar.get_height()
                        label_height = height + error + 0.01 if error > 0 else height + 0.01
                        ax.text(bar.get_x() + bar.get_width()/2., label_height,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Customize the subplot
            ax.set_title(f'{quad.upper()}', fontweight='bold')
            ax.set_xlabel('Architecture')
            ax.set_ylabel('Score')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(architectures, rotation=45, ha='right')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
            
            # # Set y-axis limits based on data
            # max_val = max([max(values) if values else 0 for values in [values]]) if values else 1
            # ax.set_ylim(0, min(max_val * 1.2, 1.1))
            ax.set_ylim(0, 0.7)
        
        # Hide unused subplots
        for idx in range(len(quadrants), len(axes)):
            axes[idx].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = output_dir / f'{category_name.lower()}_metrics_by_quad_arch.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
    
    # Create a comprehensive comparison plot
    print("\nCreating comprehensive comparison plot...")
    create_comprehensive_plot(mean_df, std_df, output_dir)

def create_comprehensive_plot(mean_df, std_df, output_dir):
    """Create a comprehensive plot comparing all Dice and Jaccard metrics."""
    
    # Select key metrics for comprehensive view
    key_metrics = [
        'Eval_Dice_Mean',
        'Eval_Jaccard_Mean',
        'Postprocess_Overall_Dice',
        'Postprocess_Overall_Jaccard',
        'Tunnel_Dice',
        'Tunnel_Jaccard'
    ]
    
    # Available metrics in the dataframe
    available_metrics = [m for m in key_metrics if m in mean_df.columns]
    
    if not available_metrics:
        print("No key metrics found for comprehensive plot")
        return
    
    # Create a large comparison plot
    n_metrics = len(available_metrics)
    cols = 3
    rows = int(np.ceil(n_metrics / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
    fig.suptitle('Comprehensive Performance Comparison: Dice and Jaccard Metrics', 
                 fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = [axes] if n_metrics == 1 else axes
    else:
        axes = axes.flatten()
    
    quadrants = sorted(mean_df['Quad'].unique())
    architectures = sorted(mean_df['Architecture'].unique())
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        # Create grouped bar plot
        x = np.arange(len(quadrants))
        width = 0.8 / len(architectures)
        
        for i, arch in enumerate(architectures):
            values = []
            errors = []
            
            for quad in quadrants:
                # Get mean data
                mean_data = mean_df[(mean_df['Quad'] == quad) & (mean_df['Architecture'] == arch)]
                
                if not mean_data.empty:
                    value = mean_data[metric].iloc[0]
                    values.append(value if pd.notna(value) else 0)
                    
                    # Get std data
                    std_data = std_df[(std_df['Quad'] == quad) & (std_df['Architecture'] == arch)] if not std_df.empty else pd.DataFrame()
                    if not std_data.empty and metric in std_data.columns:
                        std_value = std_data[metric].iloc[0]
                        errors.append(std_value if pd.notna(std_value) else 0)
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Create bars
            bars = ax.bar(x + i * width - width * (len(architectures) - 1) / 2, 
                         values, width, label=arch, alpha=0.8,
                         yerr=errors if any(errors) else None, capsize=2)
            
            # Add value labels
            for bar, value, error in zip(bars, values, errors):
                if value > 0:
                    height = bar.get_height()
                    label_height = height + error + 0.01 if error > 0 else height + 0.01
                    ax.text(bar.get_x() + bar.get_width()/2., label_height,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(metric.replace('_', ' '), fontweight='bold')
        ax.set_xlabel('Quadrant')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(quadrants)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits based on data
        if values:
            max_val = max([v for v in values if v > 0] + [0])
            ax.set_ylim(0, min(max_val * 1.3, 1.1))
    
    # Remove empty subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plot_path = output_dir / 'comprehensive_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive plot: {plot_path}")

def print_summary_statistics(mean_df, std_df):
    """Print summary statistics for the data."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"Total records (mean): {len(mean_df)}")
    print(f"Total records (std): {len(std_df)}")
    print(f"Quadrants: {sorted(mean_df['Quad'].unique())}")
    print(f"Architectures: {sorted(mean_df['Architecture'].unique())}")
    
    # Print mean performance by architecture
    print("\nMean Performance by Architecture:")
    dice_jaccard_cols = [col for col in mean_df.columns if 'dice' in col.lower() or 'jaccard' in col.lower()]
    numeric_cols = [col for col in dice_jaccard_cols if mean_df[col].dtype in ['float64', 'int64']]
    
    if numeric_cols:
        summary = mean_df.groupby('Architecture')[numeric_cols].mean()
        print(summary.round(3))
        
    # Print mean performance by quadrant
    print("\nMean Performance by Quadrant:")
    if numeric_cols:
        summary = mean_df.groupby('Quad')[numeric_cols].mean()
        print(summary.round(3))

def main():
    """Main function to run the analysis using parsed arguments."""
    import argparse

    parser = argparse.ArgumentParser(description='Create performance plots from a CSV file.')
    parser.add_argument('csv_path', nargs='?', help='Path to the CSV file')
    parser.add_argument('-o', '--output-dir', default='./plots', help='Directory to save the plots')
    args = parser.parse_args()

    csv_path = args.csv_path
    if not csv_path:
        # fallback to interactive prompt if no argument provided
        csv_path = input("Enter the path to your CSV file: ").strip()

    csv_path = Path(csv_path)
    output_dir = args.output_dir

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return

    try:
        # Create the plots
        create_performance_plots(csv_path, output_dir=output_dir)

        # Read data for summary
        df = pd.read_csv(csv_path)
        mean_df = df[~df['Quad'].str.contains('std', na=False)].copy()
        std_df = df[df['Quad'].str.contains('std', na=False)].copy()
        std_df['Quad'] = std_df['Quad'].str.replace('std', '')

        print_summary_statistics(mean_df, std_df)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Check the '{output_dir}' directory for generated visualizations.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()