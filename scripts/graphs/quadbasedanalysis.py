import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_performance_plots(csv_path, output_dir="./plots"):
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

    # Rename architectures
    architecture_name_mapping = {
        "AnisotropicUNet3D-d2-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-D2",
        "AnisotropicUNet3D-d3-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-D3",
        "AnisotropicUNet3D-d4-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-D4",
        "AnisotropicUNet3D-d5-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-D5",
        "AnisotropicUNet3D-d6-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-D6",
        "Anisotropicunet3D-D2-Hk(3-3-3)-Dk(1-2-2)": "AnisoUNet-D2",
        "Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)": "AnisoUNet-D3",
        "Anisotropicunet3D-D4-Hk(3-3-3)-Dk(1-2-2)": "AnisoUNet-D4",
        "Anisotropicunet3D-D5-Hk(3-3-3)-Dk(1-2-2)": "AnisoUNet-D5",
        "Anisotropicunet3D-D6-Hk(3-3-3)-Dk(1-2-2)": "AnisoUNet-D6",
        "UNet3d-BasicUNet": "BasicUNet3D",
        "AnisotropicUNet3DSE-d2-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-SE",
        "AnisotropicUNet3DCSAM-d2-hk(3-3-3)-dk(1-2-2)": "AnisoUNet-CSAM",
        "AnisotropicUSENet-d2-hk(3-3-3)-dk(1-2-2)": "AnisoUSENet",
        "Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)": "AnisoUNet-D3-2D",
        "Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)-Csam": "AnisoUNet-CSAM-2D",
        "Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)-Usenet-Sf16": "AnisoUSENet-2D",
        "Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Csam": "AnisoUNet-CSAM-3D",
        "Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Usenet-Sf16": "AnisoUSENet-3D",
        "Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Usenet-Sf16": "AnisoUSENet-3D",
        "Anisotropicunet3D-D3-Hk(1-3-3)-Dk(1-2-2)-Usenet": "AnisoUSENet-2D",
        "Anisotropicunet3D-D3-Hk(3-3-3)-Dk(1-2-2)-Usenet": "AnisoUSENet-3D",
    }
    df["Architecture"] = df["Architecture"].replace(architecture_name_mapping)
    print(f"Found architectures: {sorted(df['Architecture'].unique())}")
    print(f"Data loaded: {df.shape}")

    # Define key Dice and Jaccard metrics to plot
    metrics = {
        "Eval_Dice_Mean": "Evaluation Dice",
        "Eval_Jaccard_Mean": "Evaluation Jaccard",
        "Postprocess_Overall_Dice": "Postprocess Overall Dice",
        "Postprocess_Overall_Jaccard": "Postprocess Overall Jaccard",
        "Postprocess_Matched_Dice": "Postprocessed Matched Tunnels Dice",
        "Postprocess_Matched_Jaccard": "Postprocessed Matched Tunnels Jaccard",
        "Postprocess_Clean_Matched_Dice": "Postprocessed Cleanly Matched Tunnels Dice",
        "Postprocess_Clean_Matched_Jaccard": "Postprocessed Cleanly Matched Tunnels Jaccard",
        "Tunnel_Dice": "Tunnel Dice",
        "Tunnel_Jaccard": "Tunnel Jaccard",
    }

    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in df.columns}

    if not available_metrics:
        print("No metrics found!")
        return

    print(f"Creating plots for {len(available_metrics)} metrics")

    for metric_col, metric_name in available_metrics.items():
        std_col = f"{metric_col}_Std"
        if std_col not in df.columns:
            print(f"Warning: {std_col} not found, using zero errors")
            df[std_col] = 0

        data = df.copy()

        quadrants = sorted(data["Quad"].dropna().unique())
        architectures = sorted(data["Architecture"].dropna().unique())

        print(
            f"\nCreating plot for metric: {metric_name} ({metric_col}) across quadrants: {quadrants}"
        )

        if not quadrants:
            print("No quadrant information found in 'Quad' column, skipping.")
            continue

        # The user guarantees exactly 4 quadrants with filled values -> use a 2x2 layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        fig.suptitle(
            f"{metric_name} by Architecture and Quadrant",
            fontsize=16,
            fontweight="bold",
        )

        std_col = f"{metric_col}_Std"
        has_std_col = std_col in data.columns
        if not has_std_col:
            print(
                f"Warning: std column '{std_col}' not found in dataframe; error bars will be zero."
            )

        # For each quadrant, plot a grouped bar of architectures for this single metric
        for idx, quad in enumerate(quadrants):
            ax = axes[idx]

            quad_data = data[data["Quad"] == quad]

            x_pos = np.arange(len(architectures))
            values = []
            errors = []

            for arch in architectures:
                arch_rows = quad_data[quad_data["Architecture"] == arch]
                if not arch_rows.empty and metric_col in arch_rows.columns:
                    # if multiple rows exist for this (quad,arch) take the mean
                    val = pd.to_numeric(arch_rows[metric_col], errors="coerce").mean(
                        skipna=True
                    )
                    values.append(float(val) if pd.notna(val) else 0.0)
                else:
                    values.append(0.0)

                if has_std_col and not arch_rows.empty and std_col in arch_rows.columns:
                    stdv = pd.to_numeric(arch_rows[std_col], errors="coerce").mean(
                        skipna=True
                    )
                    errors.append(float(stdv) if pd.notna(stdv) else 0.0)
                else:
                    errors.append(0.0)

            # Draw bars with optional error bars
            show_errors = any(e > 0 for e in errors)
            bars = ax.bar(
                x_pos,
                values,
                0.7,
                alpha=0.8,
                yerr=errors if show_errors else None,
                capsize=3,
                label=metric_name,
            )

            # Label bars
            for bar, val, err in zip(bars, values, errors):
                if val > 0:
                    h = bar.get_height()
                    label_y = h + err + 0.01 if err > 0 else h + 0.01
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        label_y,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            ax.set_title(str(quad).upper(), fontweight="bold")
            ax.set_xlabel("Architecture")
            ax.set_ylabel("Score")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(architectures, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

            ax.set_ylim(0.0, 0.8)

        plt.tight_layout()
        filename = metric_col.replace(" ", "_").lower()
        plot_path = output_dir / f"{filename}_by_quad_arch.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")

        # Provide mean_df/std_df for downstream functions: do not split rows, keep mean_df as full data
        # and provide an empty std_df (downstream functions handle empty std_df).
        mean_df = data.copy()
        std_df = pd.DataFrame()

    # Create a comprehensive comparison plot
    print("\nCreating comprehensive plot...")
    create_comprehensive_plot(mean_df, std_df, output_dir)


def create_comprehensive_plot(mean_df, std_df, output_dir):
    """Create a comprehensive plot comparing all Dice and Jaccard metrics."""

    # Select key metrics for comprehensive view
    key_metrics = [
        "Eval_Dice_Mean",
        "Eval_Jaccard_Mean",
        "Postprocess_Overall_Dice",
        "Postprocess_Overall_Jaccard",
        "Tunnel_Dice",
        "Tunnel_Jaccard",
    ]

    # Filter available metrics (same logic as main function)
    available_metrics = {
        k: k.replace("_", " ") for k in key_metrics if k in mean_df.columns
    }

    if not available_metrics:
        print("No key metrics found for comprehensive plot")
        return

    # Extract quadrants and architectures (same logic as main function)
    data = mean_df.copy()
    quadrants = sorted(data["Quad"].dropna().unique())
    architectures = sorted(data["Architecture"].dropna().unique())

    print(f"Creating comprehensive plot with quadrants: {quadrants}")
    print(f"Architectures: {architectures}")

    if not quadrants:
        print(
            "No quadrant information found in 'Quad' column, skipping comprehensive plot."
        )
        return

    # Create a large comparison plot
    n_metrics = len(available_metrics)
    cols = 3
    rows = int(np.ceil(n_metrics / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
    fig.suptitle(
        "Comprehensive Performance Comparison: Dice and Jaccard Metrics",
        fontsize=16,
        fontweight="bold",
    )

    if rows == 1:
        axes = [axes] if n_metrics == 1 else axes
    else:
        axes = axes.flatten()

    for idx, (metric_col, metric_name) in enumerate(available_metrics.items()):
        ax = axes[idx]

        # Check for std column (same logic as main function)
        std_col = f"{metric_col}_Std"
        has_std_col = std_col in data.columns
        if not has_std_col:
            print(
                f"Warning: std column '{std_col}' not found; error bars will be zero."
            )

        # Create grouped bar plot
        x = np.arange(len(quadrants))
        width = 0.8 / len(architectures)

        # Track the maximum value across all architectures for this metric so ylim fits all bars
        metric_max = 0.0
        for i, arch in enumerate(architectures):
            values = []
            errors = []

            for quad in quadrants:
                # Get data for this quad and architecture (same logic as main function)
                quad_arch_data = data[
                    (data["Quad"] == quad) & (data["Architecture"] == arch)
                ]

                if not quad_arch_data.empty and metric_col in quad_arch_data.columns:
                    # If multiple rows exist for this (quad,arch) take the mean
                    val = pd.to_numeric(
                        quad_arch_data[metric_col], errors="coerce"
                    ).mean(skipna=True)
                    values.append(float(val) if pd.notna(val) else 0.0)
                else:
                    values.append(0.0)

                # Handle std data (same logic as main function)
                if (
                    has_std_col
                    and not quad_arch_data.empty
                    and std_col in quad_arch_data.columns
                ):
                    stdv = pd.to_numeric(quad_arch_data[std_col], errors="coerce").mean(
                        skipna=True
                    )
                    errors.append(float(stdv) if pd.notna(stdv) else 0.0)
                else:
                    errors.append(0.0)

            # Update the running metric max with this architecture's highest value
            if any(v > 0 for v in values):
                arch_max = max(values)
                if arch_max > metric_max:
                    metric_max = arch_max

            # Create bars with error bars if available
            show_errors = any(e > 0 for e in errors)
            bars = ax.bar(
                x + i * width - width * (len(architectures) - 1) / 2,
                values,
                width,
                label=arch,
                alpha=0.8,
                yerr=errors if show_errors else None,
                capsize=2,
            )

            # Add value labels (same logic as main function)
            for bar, value, error in zip(bars, values, errors):
                if value > 0:
                    height = bar.get_height()
                    label_height = height + error + 0.01 if error > 0 else height + 0.01
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        label_height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_title(metric_name, fontweight="bold")
        ax.set_xlabel("Quadrant")
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [str(q).upper() for q in quadrants]
        )  # Make consistent with main function
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits based on the maximum value observed across all architectures for this metric
        max_val = metric_max if metric_max > 0 else 1
        ax.set_ylim(0, max_val * 1.4)

    # Remove empty subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plot_path = output_dir / "comprehensive_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comprehensive plot: {plot_path}")


def print_summary_statistics(mean_df, std_df):
    """Print summary statistics for the data."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"Total records (mean): {len(mean_df)}")
    print(f"Total records (std): {len(std_df)}")
    print(f"Quadrants: {sorted(mean_df['Quad'].unique())}")
    print(f"Architectures: {sorted(mean_df['Architecture'].unique())}")

    # Print mean performance by architecture
    print("\nMean Performance by Architecture:")
    dice_jaccard_cols = [
        col
        for col in mean_df.columns
        if "dice" in col.lower() or "jaccard" in col.lower()
    ]
    numeric_cols = [
        col for col in dice_jaccard_cols if mean_df[col].dtype in ["float64", "int64"]
    ]

    if numeric_cols:
        summary = mean_df.groupby("Architecture")[numeric_cols].mean()
        print(summary.round(3))

    # Print mean performance by quadrant
    print("\nMean Performance by Quadrant:")
    if numeric_cols:
        summary = mean_df.groupby("Quad")[numeric_cols].mean()
        print(summary.round(3))


def main():
    """Main function to run the analysis using parsed arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create performance plots from a CSV file."
    )
    parser.add_argument("csv_path", nargs="?", help="Path to the CSV file")
    parser.add_argument(
        "-o", "--output-dir", default="./plots", help="Directory to save the plots"
    )
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
        mean_df = df[~df["Quad"].str.contains("std", na=False)].copy()
        std_df = df[df["Quad"].str.contains("std", na=False)].copy()
        std_df["Quad"] = std_df["Quad"].str.replace("std", "")

        print_summary_statistics(mean_df, std_df)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Check the '{output_dir}' directory for generated visualizations.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
