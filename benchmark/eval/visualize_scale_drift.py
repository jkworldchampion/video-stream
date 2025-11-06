#!/usr/bin/env python3
"""
Visualize scale/shift drift from profiling results

Generates comprehensive plots showing:
- Per-frame scale/shift values for each sequence
- Cumulative drift (prefix std) over time
- Distribution comparisons between CLIP and STREAM
- Scene-level statistics
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def load_drift_data(json_path):
    """Load drift profiling results"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def plot_sequence_drift(clip_data, stream_data, output_dir, max_sequences=10):
    """
    Plot per-frame scale/shift for individual sequences
    
    Shows first N sequences with most frames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select sequences with most frames
    clip_seqs = sorted(clip_data['sequences'], key=lambda x: x['num_frames'], reverse=True)[:max_sequences]
    
    for seq_idx, clip_seq in enumerate(clip_seqs):
        seq_id = clip_seq['sequence_id']
        
        # Find matching stream sequence
        stream_seq = next((s for s in stream_data['sequences'] if s['sequence_id'] == seq_id), None)
        if not stream_seq:
            continue
        
        # Extract data
        clip_scales = [v for v in clip_seq['scale']['per_frame'] if v is not None]
        clip_shifts = [v for v in clip_seq['shift']['per_frame'] if v is not None]
        clip_scale_prefix_std = clip_seq['scale']['prefix_std']
        clip_shift_prefix_std = clip_seq['shift']['prefix_std']
        
        stream_scales = [v for v in stream_seq['scale']['per_frame'] if v is not None]
        stream_shifts = [v for v in stream_seq['shift']['per_frame'] if v is not None]
        stream_scale_prefix_std = stream_seq['scale']['prefix_std']
        stream_shift_prefix_std = stream_seq['shift']['prefix_std']
        
        if not clip_scales or not stream_scales:
            continue
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Scale/Shift Drift: {seq_id}', fontsize=16, fontweight='bold')
        
        # Row 1: Per-frame scale and shift values
        ax1 = fig.add_subplot(gs[0, 0])
        frame_indices = list(range(len(clip_scales)))
        ax1.plot(frame_indices, clip_scales, 'o-', label='CLIP', alpha=0.7, markersize=4, linewidth=1.5)
        ax1.plot(frame_indices, stream_scales, 's-', label='STREAM', alpha=0.7, markersize=4, linewidth=1.5)
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Scale')
        ax1.set_title('Per-Frame Scale Values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(frame_indices, clip_shifts, 'o-', label='CLIP', alpha=0.7, markersize=4, linewidth=1.5)
        ax2.plot(frame_indices, stream_shifts, 's-', label='STREAM', alpha=0.7, markersize=4, linewidth=1.5)
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Shift')
        ax2.set_title('Per-Frame Shift Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Row 2: Cumulative drift (prefix std)
        ax3 = fig.add_subplot(gs[1, 0])
        prefix_indices = list(range(1, len(clip_scale_prefix_std) + 1))
        ax3.plot(prefix_indices, clip_scale_prefix_std, 'o-', label='CLIP', linewidth=2, markersize=5)
        ax3.plot(prefix_indices, stream_scale_prefix_std, 's-', label='STREAM', linewidth=2, markersize=5)
        ax3.set_xlabel('Number of Frames')
        ax3.set_ylabel('Cumulative Scale Std')
        ax3.set_title('Scale Drift Over Time (Lower = More Stable)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add final values as text
        clip_final = clip_scale_prefix_std[-1] if clip_scale_prefix_std else 0
        stream_final = stream_scale_prefix_std[-1] if stream_scale_prefix_std else 0
        ax3.text(0.05, 0.95, f'CLIP final: {clip_final:.4f}\nSTREAM final: {stream_final:.4f}',
                 transform=ax3.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(prefix_indices, clip_shift_prefix_std, 'o-', label='CLIP', linewidth=2, markersize=5)
        ax4.plot(prefix_indices, stream_shift_prefix_std, 's-', label='STREAM', linewidth=2, markersize=5)
        ax4.set_xlabel('Number of Frames')
        ax4.set_ylabel('Cumulative Shift Std')
        ax4.set_title('Shift Drift Over Time (Lower = More Stable)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add final values
        clip_final_shift = clip_shift_prefix_std[-1] if clip_shift_prefix_std else 0
        stream_final_shift = stream_shift_prefix_std[-1] if stream_shift_prefix_std else 0
        ax4.text(0.05, 0.95, f'CLIP final: {clip_final_shift:.4f}\nSTREAM final: {stream_final_shift:.4f}',
                 transform=ax4.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Row 3: Distributions
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(clip_scales, bins=30, alpha=0.6, label='CLIP', color='blue', edgecolor='black')
        ax5.hist(stream_scales, bins=30, alpha=0.6, label='STREAM', color='orange', edgecolor='black')
        ax5.axvline(np.mean(clip_scales), color='blue', linestyle='--', linewidth=2, label=f'CLIP mean: {np.mean(clip_scales):.3f}')
        ax5.axvline(np.mean(stream_scales), color='orange', linestyle='--', linewidth=2, label=f'STREAM mean: {np.mean(stream_scales):.3f}')
        ax5.set_xlabel('Scale Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Scale Distribution')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(clip_shifts, bins=30, alpha=0.6, label='CLIP', color='blue', edgecolor='black')
        ax6.hist(stream_shifts, bins=30, alpha=0.6, label='STREAM', color='orange', edgecolor='black')
        ax6.axvline(np.mean(clip_shifts), color='blue', linestyle='--', linewidth=2, label=f'CLIP mean: {np.mean(clip_shifts):.3f}')
        ax6.axvline(np.mean(stream_shifts), color='orange', linestyle='--', linewidth=2, label=f'STREAM mean: {np.mean(stream_shifts):.3f}')
        ax6.set_xlabel('Shift Value')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Shift Distribution')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Save
        output_path = os.path.join(output_dir, f'sequence_{seq_idx:02d}_{seq_id}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_path}")


def plot_overall_comparison(clip_data, stream_data, output_path):
    """
    Overall comparison across all sequences
    """
    # Extract final drift values for all sequences
    clip_scale_stds = []
    clip_shift_stds = []
    stream_scale_stds = []
    stream_shift_stds = []
    
    for seq in clip_data['sequences']:
        if seq['scale']['prefix_std'] and seq['scale']['prefix_std'][-1] is not None:
            clip_scale_stds.append(seq['scale']['prefix_std'][-1])
        if seq['shift']['prefix_std'] and seq['shift']['prefix_std'][-1] is not None:
            clip_shift_stds.append(seq['shift']['prefix_std'][-1])
    
    for seq in stream_data['sequences']:
        if seq['scale']['prefix_std'] and seq['scale']['prefix_std'][-1] is not None:
            stream_scale_stds.append(seq['scale']['prefix_std'][-1])
        if seq['shift']['prefix_std'] and seq['shift']['prefix_std'][-1] is not None:
            stream_shift_stds.append(seq['shift']['prefix_std'][-1])
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CLIP vs STREAM: Overall Drift Comparison', fontsize=18, fontweight='bold')
    
    # Row 1: Scale
    # Box plot
    ax = axes[0, 0]
    bp = ax.boxplot([clip_scale_stds, stream_scale_stds], labels=['CLIP', 'STREAM'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Final Scale Std (per sequence)')
    ax.set_title('Scale Drift Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    clip_mean = np.mean(clip_scale_stds)
    stream_mean = np.mean(stream_scale_stds)
    ax.text(0.5, 0.95, f'CLIP mean: {clip_mean:.4f}\nSTREAM mean: {stream_mean:.4f}\nDiff: {stream_mean-clip_mean:.4f}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Histogram
    ax = axes[0, 1]
    ax.hist(clip_scale_stds, bins=30, alpha=0.6, label='CLIP', color='blue', edgecolor='black')
    ax.hist(stream_scale_stds, bins=30, alpha=0.6, label='STREAM', color='red', edgecolor='black')
    ax.axvline(clip_mean, color='blue', linestyle='--', linewidth=2)
    ax.axvline(stream_mean, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Final Scale Std')
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Scale Drift Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Scatter plot
    ax = axes[0, 2]
    min_len = min(len(clip_scale_stds), len(stream_scale_stds))
    ax.scatter(clip_scale_stds[:min_len], stream_scale_stds[:min_len], alpha=0.6, s=50)
    max_val = max(max(clip_scale_stds), max(stream_scale_stds))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('CLIP Scale Std')
    ax.set_ylabel('STREAM Scale Std')
    ax.set_title('Scale Drift: CLIP vs STREAM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add quadrant info
    better_clip = sum(1 for i in range(min_len) if stream_scale_stds[i] < clip_scale_stds[i])
    ax.text(0.05, 0.95, f'STREAM better: {better_clip}/{min_len}\nCLIP better: {min_len-better_clip}/{min_len}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Row 2: Shift (same structure)
    ax = axes[1, 0]
    bp = ax.boxplot([clip_shift_stds, stream_shift_stds], labels=['CLIP', 'STREAM'],
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.set_ylabel('Final Shift Std (per sequence)')
    ax.set_title('Shift Drift Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    clip_shift_mean = np.mean(clip_shift_stds)
    stream_shift_mean = np.mean(stream_shift_stds)
    ax.text(0.5, 0.95, f'CLIP mean: {clip_shift_mean:.4f}\nSTREAM mean: {stream_shift_mean:.4f}\nDiff: {stream_shift_mean-clip_shift_mean:.4f}',
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax = axes[1, 1]
    ax.hist(clip_shift_stds, bins=30, alpha=0.6, label='CLIP', color='blue', edgecolor='black')
    ax.hist(stream_shift_stds, bins=30, alpha=0.6, label='STREAM', color='red', edgecolor='black')
    ax.axvline(clip_shift_mean, color='blue', linestyle='--', linewidth=2)
    ax.axvline(stream_shift_mean, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Final Shift Std')
    ax.set_ylabel('Number of Sequences')
    ax.set_title('Shift Drift Histogram')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 2]
    min_len = min(len(clip_shift_stds), len(stream_shift_stds))
    ax.scatter(clip_shift_stds[:min_len], stream_shift_stds[:min_len], alpha=0.6, s=50)
    max_val = max(max(clip_shift_stds), max(stream_shift_stds))
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x')
    ax.set_xlabel('CLIP Shift Std')
    ax.set_ylabel('STREAM Shift Std')
    ax.set_title('Shift Drift: CLIP vs STREAM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    better_clip_shift = sum(1 for i in range(min_len) if stream_shift_stds[i] < clip_shift_stds[i])
    ax.text(0.05, 0.95, f'STREAM better: {better_clip_shift}/{min_len}\nCLIP better: {min_len-better_clip_shift}/{min_len}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved overall comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize scale/shift drift from profiling results')
    parser.add_argument('--clip-json', required=True, help='CLIP drift profiling JSON')
    parser.add_argument('--stream-json', required=True, help='STREAM drift profiling JSON')
    parser.add_argument('--output-dir', default='./drift_plots', help='Output directory for plots')
    parser.add_argument('--max-sequences', type=int, default=10, help='Number of sequences to plot individually')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Visualizing Scale/Shift Drift")
    print("=" * 80)
    print(f"CLIP JSON: {args.clip_json}")
    print(f"STREAM JSON: {args.stream_json}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    clip_data = load_drift_data(args.clip_json)
    stream_data = load_drift_data(args.stream_json)
    print(f"✓ CLIP: {clip_data['summary']['num_sequences']} sequences")
    print(f"✓ STREAM: {stream_data['summary']['num_sequences']} sequences")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot individual sequences
    print(f"\nGenerating individual sequence plots (top {args.max_sequences})...")
    sequence_dir = os.path.join(args.output_dir, 'sequences')
    plot_sequence_drift(clip_data, stream_data, sequence_dir, args.max_sequences)
    
    # Plot overall comparison
    print("\nGenerating overall comparison...")
    overall_path = os.path.join(args.output_dir, 'overall_comparison.png')
    plot_overall_comparison(clip_data, stream_data, overall_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Summary Statistics")
    print("=" * 80)
    print("\nCLIP:")
    print(f"  Scale Drift - Mean: {clip_data['summary']['scale_std_mean']:.4f}, Median: {clip_data['summary']['scale_std_median']:.4f}")
    print(f"  Shift Drift - Mean: {clip_data['summary']['shift_std_mean']:.4f}, Median: {clip_data['summary']['shift_std_median']:.4f}")
    
    print("\nSTREAM:")
    print(f"  Scale Drift - Mean: {stream_data['summary']['scale_std_mean']:.4f}, Median: {stream_data['summary']['scale_std_median']:.4f}")
    print(f"  Shift Drift - Mean: {stream_data['summary']['shift_std_mean']:.4f}, Median: {stream_data['summary']['shift_std_median']:.4f}")
    
    print("\nDifference (STREAM - CLIP):")
    scale_diff = stream_data['summary']['scale_std_mean'] - clip_data['summary']['scale_std_mean']
    shift_diff = stream_data['summary']['shift_std_mean'] - clip_data['summary']['shift_std_mean']
    print(f"  Scale Drift: {scale_diff:+.4f} ({'STREAM more stable' if scale_diff < 0 else 'CLIP more stable'})")
    print(f"  Shift Drift: {shift_diff:+.4f} ({'STREAM more stable' if shift_diff < 0 else 'CLIP more stable'})")
    
    print("\n" + "=" * 80)
    print("✅ Visualization Complete!")
    print("=" * 80)
    print(f"\nPlots saved to: {args.output_dir}")
    print(f"  - Overall comparison: {overall_path}")
    print(f"  - Individual sequences: {sequence_dir}/")


if __name__ == '__main__':
    main()
