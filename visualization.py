import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def plot_loss_curves(history, dataset_name, epochs, save_dir='graphs', save_path=None):
    """
    Generate comprehensive loss visualization with 6 subplots.
    
    Args:
        history: Dictionary containing loss history with keys:
                 'bin_total', 'bin_intra', 'bin_inter', 'diversity', 'hardness', 'student'
        dataset_name: Name of the dataset for title and filename
        epochs: Total number of training epochs
        save_dir: Directory to save the graphs (default: 'graphs')
        save_path: Optional custom path to save the figure. If None, uses default naming.
    
    Returns:
        fig: Matplotlib figure object
        filename: Path where the figure was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Losses - {dataset_name.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')

    # 1. Total Bin Loss
    axs[0, 0].plot(history['bin_total'], color='blue', linewidth=2, alpha=0.8)
    axs[0, 0].set_title('Total Bin Loss', fontsize=12, fontweight='bold')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].set_xlim(0, epochs)

    # 2. Intra-Bin Loss
    axs[0, 1].plot(history['bin_intra'], color='red', linewidth=2, alpha=0.8)
    axs[0, 1].set_title('Intra-Bin Loss (Minimize Variance)', fontsize=12, fontweight='bold')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].set_xlim(0, epochs)

    # 3. Inter-Bin Loss (as distance - negated)
    inter_distance = [-x for x in history['bin_inter']]
    axs[0, 2].plot(inter_distance, color='green', linewidth=2, alpha=0.8)
    axs[0, 2].set_title('Inter-Bin Distance (Maximize Separation)', fontsize=12, fontweight='bold')
    axs[0, 2].set_xlabel('Epoch')
    axs[0, 2].set_ylabel('Distance')
    axs[0, 2].grid(True, alpha=0.3)
    axs[0, 2].set_xlim(0, epochs)

    # 4. Diversity Loss
    axs[1, 0].plot(history['diversity'], color='orange', linewidth=2, alpha=0.8)
    axs[1, 0].set_title('Diversity Loss (Entropy)', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_xlim(0, epochs)

    # 5. Hardness Loss
    axs[1, 1].plot(history['hardness'], color='brown', linewidth=2, alpha=0.8)
    axs[1, 1].set_title('Hardness Loss (Adversarial)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Loss')
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_xlim(0, epochs)

    # 6. Student KD Loss
    axs[1, 2].plot(history['student'], color='cyan', linewidth=2, alpha=0.8)
    axs[1, 2].set_title('Student KD Loss', fontsize=12, fontweight='bold')
    axs[1, 2].set_xlabel('Epoch')
    axs[1, 2].set_ylabel('Loss')
    axs[1, 2].grid(True, alpha=0.3)
    axs[1, 2].set_xlim(0, epochs)

    plt.tight_layout()

    # Save figure
    if save_path is None:
        save_path = os.path.join(save_dir, f'loss_curves_{dataset_name}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss curves saved to '{save_path}'")
    
    return fig, save_path


def plot_performance_metrics(history, dataset_name, teacher_acc, warmup_acc, epochs, save_dir='graphs', save_path=None):
    """
    Generate performance metrics visualization (accuracy, agreement, coverage).
    
    Args:
        history: Dictionary containing metrics with keys:
                 'test_acc', 'agreement', 'coverage'
        dataset_name: Name of the dataset for title and filename
        teacher_acc: Teacher's test accuracy for reference line
        warmup_acc: Student's warmup test accuracy for reference line
        epochs: Total number of training epochs
        save_dir: Directory to save the graphs (default: 'graphs')
        save_path: Optional custom path to save the figure
    
    Returns:
        fig: Matplotlib figure object
        filename: Path where the figure was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Performance Metrics - {dataset_name.replace("_", " ").title()}', 
                 fontsize=16, fontweight='bold')
    
    # Create epoch indices for metrics (sampled every 20 epochs)
    epochs_plot = [i*20 for i in range(len(history['test_acc']))]
    
    # 1. Test Accuracy
    axs[0].plot(epochs_plot, [acc*100 for acc in history['test_acc']], 
                'b-', linewidth=2, label='Student', marker='o', markersize=4)
    axs[0].axhline(y=teacher_acc*100, color='r', linestyle='--', linewidth=2, label='Teacher')
    axs[0].axhline(y=warmup_acc*100, color='g', linestyle=':', linewidth=2, label='Warmup')
    axs[0].set_title('Test Accuracy Over Time', fontsize=12, fontweight='bold')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    axs[0].set_xlim(0, epochs)
    
    # 2. Teacher-Student Agreement
    axs[1].plot(epochs_plot, [agr*100 for agr in history['agreement']], 
                color='purple', linewidth=2, marker='o', markersize=4)
    axs[1].set_title('Teacher-Student Agreement', fontsize=12, fontweight='bold')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Agreement (%)')
    axs[1].grid(True, alpha=0.3)
    axs[1].set_xlim(0, epochs)
    
    # 3. Coverage Certification
    axs[2].plot(epochs_plot, [c*100 for c in history['coverage']], 
                color='teal', linewidth=2, marker='o', markersize=4)
    axs[2].set_title('Coverage Certification', fontsize=12, fontweight='bold')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Coverage (%)')
    axs[2].grid(True, alpha=0.3)
    axs[2].set_xlim(0, epochs)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(save_dir, f'performance_metrics_{dataset_name}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance metrics saved to '{save_path}'")
    
    return fig, save_path


def plot_combined_dashboard(history, dataset_name, teacher_acc, warmup_acc, epochs, save_dir='graphs', save_path=None):
    """
    Generate comprehensive dashboard with all metrics in one figure.
    
    Args:
        history: Dictionary containing all metrics
        dataset_name: Name of the dataset
        teacher_acc: Teacher's test accuracy
        warmup_acc: Student's warmup test accuracy
        epochs: Total number of training epochs
        save_dir: Directory to save the graphs (default: 'graphs')
        save_path: Optional custom path to save the figure
    
    Returns:
        fig: Matplotlib figure object
        filename: Path where the figure was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Training Dashboard - {dataset_name.replace("_", " ").title()}', 
                 fontsize=18, fontweight='bold')
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Bin Losses
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['bin_total'], color='blue', linewidth=2, alpha=0.8)
    ax1.set_title('Total Bin Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, epochs)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['bin_intra'], color='red', linewidth=2, alpha=0.8)
    ax2.set_title('Intra-Bin Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, epochs)
    
    ax3 = fig.add_subplot(gs[0, 2])
    inter_distance = [-x for x in history['bin_inter']]
    ax3.plot(inter_distance, color='green', linewidth=2, alpha=0.8)
    ax3.set_title('Inter-Bin Distance', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Distance')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, epochs)
    
    # Row 2: Generator Losses
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(history['diversity'], color='orange', linewidth=2, alpha=0.8)
    ax4.set_title('Diversity Loss', fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, epochs)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['hardness'], color='brown', linewidth=2, alpha=0.8)
    ax5.set_title('Hardness Loss', fontweight='bold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Loss')
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, epochs)
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(history['student'], color='cyan', linewidth=2, alpha=0.8)
    ax6.set_title('Student KD Loss', fontweight='bold')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, epochs)
    
    # Row 3: Performance Metrics
    epochs_plot = [i*20 for i in range(len(history['test_acc']))]
    
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(epochs_plot, [acc*100 for acc in history['test_acc']], 
             'b-', linewidth=2, label='Student', marker='o', markersize=3)
    ax7.axhline(y=teacher_acc*100, color='r', linestyle='--', linewidth=2, label='Teacher')
    ax7.axhline(y=warmup_acc*100, color='g', linestyle=':', linewidth=2, label='Warmup')
    ax7.set_title('Test Accuracy', fontweight='bold')
    ax7.set_xlabel('Epoch')
    ax7.set_ylabel('Accuracy (%)')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(0, epochs)
    
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(epochs_plot, [agr*100 for agr in history['agreement']], 
             color='purple', linewidth=2, marker='o', markersize=3)
    ax8.set_title('Teacher-Student Agreement', fontweight='bold')
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Agreement (%)')
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(0, epochs)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(epochs_plot, [c*100 for c in history['coverage']], 
             color='teal', linewidth=2, marker='o', markersize=3)
    ax9.set_title('Coverage Certification', fontweight='bold')
    ax9.set_xlabel('Epoch')
    ax9.set_ylabel('Coverage (%)')
    ax9.grid(True, alpha=0.3)
    ax9.set_xlim(0, epochs)
    
    # Save figure
    if save_path is None:
        save_path = os.path.join(save_dir, f'training_dashboard_{dataset_name}.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training dashboard saved to '{save_path}'")
    
    return fig, save_path


def generate_loss_report(history, dataset_name, epochs, save_dir='reports', save_path=None, report_interval=20):
    """
    Generate comprehensive text report with all losses over epochs.
    
    Args:
        history: Dictionary containing loss history with keys:
                 'bin_total', 'bin_intra', 'bin_inter', 'diversity', 'hardness', 'student',
                 'test_acc', 'agreement', 'coverage'
        dataset_name: Name of the dataset for title and filename
        epochs: Total number of training epochs
        save_dir: Directory to save the reports (default: 'reports')
        save_path: Optional custom path to save the report
        report_interval: Report losses every N epochs (default: 20)
    
    Returns:
        filename: Path where the report was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(save_dir, f'loss_report_{dataset_name}.txt')
    
    with open(save_path, 'w') as f:
        # Header
        f.write("=" * 120 + "\n")
        f.write(f"TRAINING LOSS REPORT: {dataset_name.replace('_', ' ').upper()}\n")
        f.write("=" * 120 + "\n\n")
        
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Report Interval: Every {report_interval} epoch(s)\n")
        f.write(f"Total Loss Components: 6 (Bin Total, Bin Intra, Bin Inter, Diversity, Hardness, Student)\n\n")
        
        # Section 1: Loss Evolution Over Epochs
        f.write("=" * 120 + "\n")
        f.write("SECTION 1: LOSS EVOLUTION OVER EPOCHS\n")
        f.write("=" * 120 + "\n\n")
        
        # Table header
        f.write(f"{'Epoch':<8} | {'Bin Total':<12} | {'Bin Intra':<12} | {'Bin Inter':<12} | {'Inter Dist':<12} | {'Diversity':<12} | {'Hardness':<12} | {'Student':<12}\n")
        f.write("-" * 120 + "\n")
        
        # Report losses at specified intervals
        for epoch in range(1, epochs + 1):
            if epoch % report_interval == 0 or epoch == 1:
                idx = epoch - 1
                bin_total = history['bin_total'][idx]
                bin_intra = history['bin_intra'][idx]
                bin_inter = history['bin_inter'][idx]
                inter_dist = -bin_inter  # Negated to show as distance
                diversity = history['diversity'][idx]
                hardness = history['hardness'][idx]
                student = history['student'][idx]
                
                f.write(f"{epoch:<8} | {bin_total:<12.6f} | {bin_intra:<12.6f} | {bin_inter:<12.6f} | {inter_dist:<12.6f} | {diversity:<12.6f} | {hardness:<12.6f} | {student:<12.6f}\n")
        
        # Section 2: Loss Statistics Summary
        f.write("\n" + "=" * 120 + "\n")
        f.write("SECTION 2: LOSS STATISTICS SUMMARY\n")
        f.write("=" * 120 + "\n\n")
        
        # Calculate statistics for each loss component
        loss_components = {
            'Bin Total Loss': history['bin_total'],
            'Bin Intra Loss': history['bin_intra'],
            'Bin Inter Loss': history['bin_inter'],
            'Inter-Bin Distance': [-x for x in history['bin_inter']],
            'Diversity Loss': history['diversity'],
            'Hardness Loss': history['hardness'],
            'Student KD Loss': history['student']
        }
        
        for name, values in loss_components.items():
            initial = values[0]
            final = values[-1]
            minimum = min(values)
            maximum = max(values)
            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            change = final - initial
            change_pct = (change / abs(initial) * 100) if initial != 0 else 0;
            
            f.write(f"{name}:\n")
            f.write(f"  Initial (Epoch 1):   {initial:12.6f}\n")
            f.write(f"  Final (Epoch {epochs}):    {final:12.6f}\n")
            f.write(f"  Minimum:             {minimum:12.6f}\n")
            f.write(f"  Maximum:             {maximum:12.6f}\n")
            f.write(f"  Mean:                {mean:12.6f}\n")
            f.write(f"  Std Dev:             {std:12.6f}\n")
            f.write(f"  Change:              {change:+12.6f} ({change_pct:+.2f}%)\n")
            f.write("\n")
        
        # Section 3: Performance Metrics Over Epochs
        if history['test_acc']:  # Check if performance metrics are available
            f.write("=" * 120 + "\n")
            f.write("SECTION 3: PERFORMANCE METRICS OVER EPOCHS\n")
            f.write("=" * 120 + "\n\n")
            
            f.write(f"{'Epoch':<8} | {'Test Accuracy':<15} | {'Agreement':<15} | {'Coverage':<15}\n")
            f.write("-" * 60 + "\n")
            
            for i, epoch in enumerate(range(report_interval, epochs + 1, report_interval)):
                test_acc = history['test_acc'][i] * 100
                agreement = history['agreement'][i] * 100
                coverage = history['coverage'][i] * 100
                
                f.write(f"{epoch:<8} | {test_acc:<15.2f}% | {agreement:<15.2f}% | {coverage:<15.2f}%\n")
            
            # Performance statistics
            f.write("\nPerformance Statistics:\n")
            f.write(f"  Test Accuracy:\n")
            f.write(f"    Initial:  {history['test_acc'][0]*100:.2f}%\n")
            f.write(f"    Final:    {history['test_acc'][-1]*100:.2f}%\n")
            f.write(f"    Best:     {max(history['test_acc'])*100:.2f}%\n")
            f.write(f"    Change:   {(history['test_acc'][-1] - history['test_acc'][0])*100:+.2f}%\n\n")
            
            f.write(f"  Teacher-Student Agreement:\n")
            f.write(f"    Initial:  {history['agreement'][0]*100:.2f}%\n")
            f.write(f"    Final:    {history['agreement'][-1]*100:.2f}%\n")
            f.write(f"    Best:     {max(history['agreement'])*100:.2f}%\n")
            f.write(f"    Change:   {(history['agreement'][-1] - history['agreement'][0])*100:+.2f}%\n\n")
            
            f.write(f"  Coverage Certification:\n")
            f.write(f"    Initial:  {history['coverage'][0]*100:.2f}%\n")
            f.write(f"    Final:    {history['coverage'][-1]*100:.2f}%\n")
            f.write(f"    Best:     {max(history['coverage'])*100:.2f}%\n")
            f.write(f"    Change:   {(history['coverage'][-1] - history['coverage'][0])*100:+.2f}%\n\n")
        
        # Section 4: Training Observations
        f.write("=" * 120 + "\n")
        f.write("SECTION 4: TRAINING OBSERVATIONS\n")
        f.write("=" * 120 + "\n\n")
        
        # Analyze trends
        bin_total_trend = "decreasing" if history['bin_total'][-1] < history['bin_total'][0] else "increasing"
        intra_trend = "decreasing" if history['bin_intra'][-1] < history['bin_intra'][0] else "increasing"
        inter_dist = [-x for x in history['bin_inter']]
        inter_trend = "increasing" if inter_dist[-1] > inter_dist[0] else "decreasing"
        student_trend = "decreasing" if history['student'][-1] < history['student'][0] else "increasing"
        
        f.write(f"Loss Trends:\n")
        f.write(f"  • Total Bin Loss:       {bin_total_trend.upper()}\n")
        f.write(f"  • Intra-Bin Loss:       {intra_trend.upper()} (purer bins)\n")
        f.write(f"  • Inter-Bin Distance:   {inter_trend.upper()} (better separation)\n")
        f.write(f"  • Student KD Loss:      {student_trend.upper()}\n\n")
        
        # Quality assessment
        f.write("Quality Assessment:\n")
        if intra_trend == "decreasing" and inter_trend == "increasing":
            f.write("  ✓ EXCELLENT: Bins are becoming purer (lower intra) and more separated (higher inter)\n")
        elif intra_trend == "decreasing":
            f.write("  ⚠ PARTIAL: Bins are purer but separation is not improving\n")
        elif inter_trend == "increasing":
            f.write("  ⚠ PARTIAL: Bin separation is improving but purity is not\n")
        else:
            f.write("  ✗ NEEDS IMPROVEMENT: Consider adjusting temperature schedule or loss weights\n")
        
        f.write("\n")
        
        # Footer
        f.write("=" * 120 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 120 + "\n")
    
    print(f"✅ Loss report saved to '{save_path}'")
    return save_path


def save_history(history, dataset_name, save_dir='graphs', save_path=None):
    """
    Save training history to disk.
    
    Args:
        history: Dictionary containing training metrics
        dataset_name: Name of the dataset
        save_dir: Directory to save the history (default: 'graphs')
        save_path: Optional custom path to save the history
    
    Returns:
        filename: Path where the history was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(save_dir, f'training_history_{dataset_name}.pt')
    
    torch.save(history, save_path)
    print(f"✅ Training history saved to '{save_path}'")
    
    return save_path


def generate_variance_report(history, dataset_name, num_bins, num_features, save_dir='reports', save_path=None):
    """
    Generate a comprehensive variance report grouped by feature.
    Each feature shows bins as rows and epochs as columns.
    
    Args:
        history: Dictionary containing 'variance_maps' and 'inter_variance_per_feature'
        dataset_name: Name of the dataset
        num_bins: Number of bins per feature (K)
        num_features: Number of features (F)
        save_dir: Directory to save the report (default: 'reports')
        save_path: Optional custom path to save the report
    
    Returns:
        filename: Path where the report was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    if save_path is None:
        save_path = os.path.join(save_dir, f'variance_report_{dataset_name}.txt')
    
    # Get variance data
    variance_maps = history['variance_maps']
    inter_variance_per_feature = history['inter_variance_per_feature']
    
    epochs_reported = len(variance_maps)
    epoch_indices = list(range(20, 20 * epochs_reported + 1, 20))  # Every 20 epochs
    
    with open(save_path, 'w') as f:
        # Header
        f.write("=" * 150 + "\n")
        f.write(f"BIN VARIANCE ANALYSIS REPORT: {dataset_name.replace('_', ' ').upper()}\n")
        f.write("=" * 150 + "\n\n")
        
        f.write(f"Number of Features: {num_features}\n")
        f.write(f"Number of Bins per Feature: {num_bins}\n")
        f.write(f"Epochs Analyzed: {epoch_indices}\n")
        f.write(f"Total Measurements: {epochs_reported}\n\n")
        
        f.write("VARIANCE METRICS:\n")
        f.write("  • Intra-Bin Variance: Measures how tight/pure each bin is (LOWER is better)\n")
        f.write("  • Inter-Bin Variance: Measures separation between bins per feature (HIGHER is better)\n\n")
        
        # Main content: Grouped by feature with epochs as columns
        f.write("=" * 150 + "\n")
        f.write("DETAILED VARIANCE ANALYSIS (BINS × EPOCHS)\n")
        f.write("=" * 150 + "\n\n")
        
        for f_idx in range(num_features):
            f.write("\n" + "─" * 150 + "\n")
            f.write(f"FEATURE {f_idx}\n")
            f.write("─" * 150 + "\n\n")
            
            # Inter-bin variance header (shows separation quality)
            f.write(f"INTER-BIN VARIANCE (Bin Separation Quality - HIGHER is better):\n")
            
            # Create header with epoch columns
            header = f"{'Metric':<15} |"
            for epoch in epoch_indices:
                header += f" Epoch {epoch:<5} |"
            header += f" Change   | Trend"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Inter-variance row
            inter_row = f"{'Inter-Var':<15} |"
            inter_vals = [inter_variance_per_feature[i][f_idx].item() for i in range(epochs_reported)]
            for val in inter_vals:
                inter_row += f" {val:>11.6f} |"
            
            change = inter_vals[-1] - inter_vals[0]
            if change > 0.01:
                trend = "↑ Improving"
            elif change < -0.01:
                trend = "↓ Degrading"
            else:
                trend = "→ Stable"
            
            inter_row += f" {change:>+8.6f} | {trend}"
            f.write(inter_row + "\n\n")
            
            # Intra-bin variance table (bins as rows, epochs as columns)
            f.write(f"INTRA-BIN VARIANCE (Bin Purity - LOWER is better):\n")
            
            # Create header with epoch columns
            header = f"{'Bin':<15} |"
            for epoch in epoch_indices:
                header += f" Epoch {epoch:<5} |"
            header += f" Change   | Trend"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            # Each bin as a row
            for k in range(num_bins):
                bin_row = f"{'Bin ' + str(k):<15} |"
                intra_vals = [variance_maps[i][f_idx, k].item() for i in range(epochs_reported)]
                
                for val in intra_vals:
                    bin_row += f" {val:>11.6f} |"
                
                change = intra_vals[-1] - intra_vals[0]
                if change < -0.01:
                    trend = "↓ Improving"
                elif change > 0.01:
                    trend = "↑ Degrading"
                else:
                    trend = "→ Stable"
                
                bin_row += f" {change:>+8.6f} | {trend}"
                f.write(bin_row + "\n")
            
            f.write("\n")
            
            # Summary statistics for this feature
            f.write(f"Summary Statistics for Feature {f_idx}:\n")
            
            # Inter-variance stats
            inter_initial = inter_vals[0]
            inter_final = inter_vals[-1]
            inter_max = max(inter_vals)
            inter_min = min(inter_vals)
            inter_mean = sum(inter_vals) / len(inter_vals)
            f.write(f"  Inter-Bin:  Initial={inter_initial:.6f}, Final={inter_final:.6f}, ")
            f.write(f"Min={inter_min:.6f}, Max={inter_max:.6f}, Mean={inter_mean:.6f}\n")
            
            # Intra-variance stats across all bins
            all_intra = []
            for k in range(num_bins):
                all_intra.extend([variance_maps[i][f_idx, k].item() for i in range(epochs_reported)])
            
            intra_initial_avg = sum([variance_maps[0][f_idx, k].item() for k in range(num_bins)]) / num_bins
            intra_final_avg = sum([variance_maps[-1][f_idx, k].item() for k in range(num_bins)]) / num_bins
            intra_min = min(all_intra)
            intra_max = max(all_intra)
            intra_mean = sum(all_intra) / len(all_intra)
            
            f.write(f"  Intra-Bins: Initial Avg={intra_initial_avg:.6f}, Final Avg={intra_final_avg:.6f}, ")
            f.write(f"Min={intra_min:.6f}, Max={intra_max:.6f}, Mean={intra_mean:.6f}\n\n")
        
        # Overall summary
        f.write("\n" + "=" * 150 + "\n")
        f.write("OVERALL SUMMARY\n")
        f.write("=" * 150 + "\n\n")
        
        # Count improvements
        improving_inter = 0
        degrading_inter = 0
        stable_inter = 0
        
        improving_intra = 0
        degrading_intra = 0
        stable_intra = 0
        
        for f_idx in range(num_features):
            inter_vals = [inter_variance_per_feature[i][f_idx].item() for i in range(epochs_reported)]
            inter_change = inter_vals[-1] - inter_vals[0]
            
            if inter_change > 0.01:
                improving_inter += 1
            elif inter_change < -0.01:
                degrading_inter += 1
            else:
                stable_inter += 1
            
            for k in range(num_bins):
                intra_vals = [variance_maps[i][f_idx, k].item() for i in range(epochs_reported)]
                intra_change = intra_vals[-1] - intra_vals[0]
                
                if intra_change < -0.01:
                    improving_intra += 1
                elif intra_change > 0.01:
                    degrading_intra += 1
                else:
                    stable_intra += 1
        
        f.write(f"Inter-Bin Variance Trends (across {num_features} features):\n")
        f.write(f"  ↑ Improving (better separation): {improving_inter} ({improving_inter/num_features*100:.1f}%)\n")
        f.write(f"  ↓ Degrading:                     {degrading_inter} ({degrading_inter/num_features*100:.1f}%)\n")
        f.write(f"  → Stable:                        {stable_inter} ({stable_inter/num_features*100:.1f}%)\n\n")
        
        total_bins = num_features * num_bins
        f.write(f"Intra-Bin Variance Trends (across {total_bins} bins):\n")
        f.write(f"  ↓ Improving (purer bins):        {improving_intra} ({improving_intra/total_bins*100:.1f}%)\n")
        f.write(f"  ↑ Degrading:                     {degrading_intra} ({degrading_intra/total_bins*100:.1f}%)\n")
        f.write(f"  → Stable:                        {stable_intra} ({stable_intra/total_bins*100:.1f}%)\n\n")
        
        # Quality assessment
        f.write("Quality Assessment:\n")
        if improving_inter > num_features * 0.7 and improving_intra > total_bins * 0.7:
            f.write("  ✓ EXCELLENT: Most features show improving bin separation AND bin purity\n")
        elif improving_inter > num_features * 0.5 and improving_intra > total_bins * 0.5:
            f.write("  ✓ GOOD: Majority of features show improvement in separation and purity\n")
        elif improving_inter > num_features * 0.3 or improving_intra > total_bins * 0.3:
            f.write("  ⚠ PARTIAL: Some improvement but many bins/features are stable or degrading\n")
        else:
            f.write("  ✗ NEEDS IMPROVEMENT: Consider adjusting bin learner learning rate or warmup strategy\n")
        
        f.write("\n")
        
        # Footer
        f.write("=" * 150 + "\n")
        f.write("END OF VARIANCE REPORT\n")
        f.write("=" * 150 + "\n")
    
    print(f"✅ Variance report saved to '{save_path}'")
    return save_path