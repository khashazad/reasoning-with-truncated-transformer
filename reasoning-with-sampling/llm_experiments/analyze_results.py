#!/usr/bin/env python3
"""
Comprehensive analysis of truncated model results.
Compares truncation methods and generates visualizations.
"""

import pandas as pd
import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from grader_utils.math_grader import grade_answer


def safe_grade(ans, correct_ans):
    """Safely grade an answer, returning 0 on error."""
    try:
        if pd.isna(ans):
            return 0
        return int(grade_answer(str(ans), str(correct_ans)))
    except Exception:
        return 0


def parse_filename(fname):
    """Extract parameters from filename like: truncated_ps_results_layer25_steps10_temp0.5_batch0.csv
    or baseline_results_layer25_temp0.5_batch0.csv"""
    # Try MCMC format first
    pattern = r"layer(\d+)_steps(\d+)_temp([\d.]+)_batch(\d+)"
    match = re.search(pattern, fname)
    if match:
        return {
            "layer": int(match.group(1)),
            "steps": int(match.group(2)),
            "temp": float(match.group(3)),
            "batch": int(match.group(4)),
        }
    
    # Try baseline format (no steps)
    pattern_baseline = r"layer(\d+)_temp([\d.]+)_batch(\d+)"
    match_baseline = re.search(pattern_baseline, fname)
    if match_baseline:
        return {
            "layer": int(match_baseline.group(1)),
            "steps": 0,  # No MCMC steps for baseline
            "temp": float(match_baseline.group(2)),
            "batch": int(match_baseline.group(3)),
        }
    return None


def evaluate_csv(filepath):
    """Evaluate a single CSV file and return detailed results."""
    df = pd.read_csv(filepath)
    total = len(df)
    correct = 0
    
    for i in range(total):
        correct += safe_grade(df["mcmc_answer"][i], df["correct_answer"][i])
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "filepath": str(filepath),
        "total_questions": total,
        "correct": correct,
        "accuracy": accuracy,
    }


def analyze_folder(folder_path, method_name="Unknown"):
    """Analyze all CSVs in a folder and return aggregated results."""
    folder = Path(folder_path)
    results = []
    
    for csv_file in sorted(folder.glob("**/*.csv")):
        params = parse_filename(csv_file.name)
        if params is None:
            print(f"Warning: Could not parse filename: {csv_file.name}")
            continue
        
        eval_result = evaluate_csv(csv_file)
        eval_result.update(params)
        eval_result["method"] = method_name
        eval_result["model"] = csv_file.parent.name
        results.append(eval_result)
    
    return results


def aggregate_by_layer(results):
    """Aggregate results by layer (combining batches)."""
    layer_data = defaultdict(lambda: {"correct": 0, "total": 0, "method": None, "model": None})
    
    for r in results:
        key = (r["layer"], r["method"], r["model"])
        layer_data[key]["correct"] += r["correct"]
        layer_data[key]["total"] += r["total_questions"]
        layer_data[key]["method"] = r["method"]
        layer_data[key]["model"] = r["model"]
        layer_data[key]["layer"] = r["layer"]
    
    aggregated = []
    for key, data in layer_data.items():
        accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
        aggregated.append({
            "layer": data["layer"],
            "method": data["method"],
            "model": data["model"],
            "total_questions": data["total"],
            "correct": data["correct"],
            "accuracy": accuracy,
        })
    
    return sorted(aggregated, key=lambda x: (x["method"], x["layer"]))


def print_detailed_results(results, title):
    """Print detailed results for each CSV."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)
    
    for r in results:
        layer_str = f"Layer {r['layer']}" if r['layer'] < 100 else "Full Model"
        print(f"\n  {layer_str} | Batch {r['batch']}")
        print(f"    File: {Path(r['filepath']).name}")
        print(f"    Questions: {r['total_questions']}")
        print(f"    Correct:   {r['correct']}")
        print(f"    Accuracy:  {r['accuracy']:.1%}")


def print_aggregated_results(aggregated, title):
    """Print aggregated results by layer."""
    print("\n" + "=" * 80)
    print(f" {title} - Aggregated by Layer")
    print("=" * 80)
    print(f"\n  {'Layer':<15} {'Questions':<12} {'Correct':<10} {'Accuracy':<10}")
    print("  " + "-" * 47)
    
    for r in aggregated:
        layer_str = f"Layer {r['layer']}" if r['layer'] < 100 else "Full Model"
        print(f"  {layer_str:<15} {r['total_questions']:<12} {r['correct']:<10} {r['accuracy']:.1%}")


def create_comparison_plot(all_aggregated, output_path):
    """Create comparison bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Group by method
    methods = {}
    for r in all_aggregated:
        method_key = f"{r['method']}\n({r['model']})"
        if method_key not in methods:
            methods[method_key] = []
        methods[method_key].append(r)
    
    # Prepare data for plotting
    x_labels = []
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
    
    bar_width = 0.35
    x_positions = []
    current_x = 0
    
    for method_idx, (method_name, method_results) in enumerate(methods.items()):
        method_results = sorted(method_results, key=lambda x: x['layer'])
        
        for i, r in enumerate(method_results):
            layer_str = f"L{r['layer']}" if r['layer'] < 100 else "Full"
            x_labels.append(f"{layer_str}")
            x_positions.append(current_x)
            
            color = colors[method_idx % len(colors)]
            bar = ax.bar(current_x, r['accuracy'] * 100, bar_width * 1.5, 
                        color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value label on bar
            ax.text(current_x, r['accuracy'] * 100 + 1, f"{r['accuracy']:.1%}", 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            current_x += 1
        
        current_x += 0.5  # Gap between methods
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Layer Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Truncated Model Performance Comparison\n(Power Sampling on MATH)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=0, fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = list(methods.keys())
    legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i % len(colors)], alpha=0.8) 
                     for i in range(len(legend_labels))]
    ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to: {output_path}")
    plt.close()


def create_layer_trend_plot(all_aggregated, output_path):
    """Create line plot showing accuracy trend by layer."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by method
    methods = {}
    for r in all_aggregated:
        method_key = f"{r['method']} ({r['model']})"
        if method_key not in methods:
            methods[method_key] = []
        methods[method_key].append(r)
    
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    for idx, (method_name, method_results) in enumerate(methods.items()):
        method_results = sorted(method_results, key=lambda x: x['layer'])
        
        layers = []
        accuracies = []
        for r in method_results:
            layer_val = r['layer'] if r['layer'] < 100 else 28  # Treat 100 as "full" (28 layers)
            layers.append(layer_val)
            accuracies.append(r['accuracy'] * 100)
        
        ax.plot(layers, accuracies, marker=markers[idx % len(markers)], 
               color=colors[idx % len(colors)], linewidth=2, markersize=8,
               label=method_name, alpha=0.8)
        
        # Add value labels
        for x, y in zip(layers, accuracies):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                       xytext=(0, 8), ha='center', fontsize=8)
    
    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Truncation Layer\n(Power Sampling on MATH)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved trend plot to: {output_path}")
    plt.close()


def create_summary_table(all_aggregated, output_path):
    """Create and save a summary CSV table."""
    df = pd.DataFrame(all_aggregated)
    df['accuracy_pct'] = df['accuracy'].apply(lambda x: f"{x:.1%}")
    df = df[['method', 'model', 'layer', 'total_questions', 'correct', 'accuracy_pct']]
    df.columns = ['Method', 'Model', 'Layer', 'Total Questions', 'Correct', 'Accuracy']
    df.to_csv(output_path, index=False)
    print(f"Saved summary table to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Analyze truncated model results")
    parser.add_argument("--results", type=str, default="results",
                       help="Path to results folder (MCMC + truncation + last layer)")
    parser.add_argument("--results_truncated", type=str, default="results_truncated",
                       help="Path to results_truncated folder (MCMC + truncation only)")
    parser.add_argument("--results_baseline", type=str, default="results_baseline",
                       help="Path to baseline results folder (no MCMC)")
    parser.add_argument("--output_dir", type=str, default="analysis_output",
                       help="Output directory for plots and tables")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    all_aggregated = []
    
    # Analyze results folder (MCMC + truncation + keep last layer)
    results_path = Path(args.results)
    if results_path.exists():
        print(f"\nAnalyzing: {results_path}")
        results = analyze_folder(results_path, method_name="MCMC + Trunc + Last Layer")
        if results:
            print_detailed_results(results, "Results: MCMC + Truncation + Last Layer")
            aggregated = aggregate_by_layer(results)
            print_aggregated_results(aggregated, "Results: MCMC + Truncation + Last Layer")
            all_results.extend(results)
            all_aggregated.extend(aggregated)
    else:
        print(f"Warning: {results_path} not found")
    
    # Analyze results_truncated folder (MCMC + truncation only)
    results_truncated_path = Path(args.results_truncated)
    if results_truncated_path.exists():
        print(f"\nAnalyzing: {results_truncated_path}")
        results_trunc = analyze_folder(results_truncated_path, method_name="MCMC + Truncation Only")
        if results_trunc:
            print_detailed_results(results_trunc, "Results: MCMC + Truncation Only")
            aggregated_trunc = aggregate_by_layer(results_trunc)
            print_aggregated_results(aggregated_trunc, "Results: MCMC + Truncation Only")
            all_results.extend(results_trunc)
            all_aggregated.extend(aggregated_trunc)
    else:
        print(f"Warning: {results_truncated_path} not found")
    
    # Analyze baseline results folder (no MCMC)
    results_baseline_path = Path(args.results_baseline)
    if results_baseline_path.exists():
        print(f"\nAnalyzing: {results_baseline_path}")
        results_baseline = analyze_folder(results_baseline_path, method_name="Baseline (No MCMC)")
        if results_baseline:
            print_detailed_results(results_baseline, "Results: Baseline (No MCMC)")
            aggregated_baseline = aggregate_by_layer(results_baseline)
            print_aggregated_results(aggregated_baseline, "Results: Baseline (No MCMC)")
            all_results.extend(results_baseline)
            all_aggregated.extend(aggregated_baseline)
    else:
        print(f"Note: {results_baseline_path} not found (run baseline experiments first)")
    
    # Generate visualizations
    if all_aggregated:
        print("\n" + "=" * 80)
        print(" Generating Visualizations")
        print("=" * 80)
        
        create_comparison_plot(all_aggregated, output_dir / "comparison_bar_chart.png")
        create_layer_trend_plot(all_aggregated, output_dir / "layer_trend_plot.png")
        summary_df = create_summary_table(all_aggregated, output_dir / "summary_table.csv")
        
        print("\n" + "=" * 80)
        print(" Summary Table")
        print("=" * 80)
        print(summary_df.to_string(index=False))
    else:
        print("\nNo results found to analyze.")


if __name__ == "__main__":
    main()


