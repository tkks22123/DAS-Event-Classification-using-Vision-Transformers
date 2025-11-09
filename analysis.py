import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import json
import numpy as np
from pathlib import Path


def perform_ablation_study(trainer, test_loader, class_names):
    """
    Perform comprehensive ablation study and model analysis

    Args:
        trainer: Trained DASTrainer instance
        test_loader: DataLoader for test set
        class_names: List of class names for labeling

    Returns:
        report_df: DataFrame containing detailed classification report
    """
    print("🔬 Performing Ablation Study and Model Analysis...")

    # Ensure output directory exists
    output_dir = Path("training_results")
    output_dir.mkdir(exist_ok=True)

    # 1. Evaluate model performance on test set
    print("   - Evaluating model on test set...")
    test_acc, test_loss, predictions, targets = trainer.evaluate_model(test_loader)

    # 2. Generate confusion matrix for error analysis
    print("   - Generating confusion matrix...")
    cm = confusion_matrix(targets, predictions)

    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Model Prediction Analysis', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Generate detailed classification report
    print("   - Generating classification report...")
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Save classification report
    report_df.to_csv(output_dir / 'detailed_classification_report.csv')

    # 4. Create performance visualization by class
    print("   - Creating per-class performance visualization...")
    plot_per_class_performance(report_df, class_names, output_dir)

    # 5. Analyze training history for insights
    print("   - Analyzing training history...")
    analyze_training_history(trainer, output_dir)

    print("✅ Ablation study completed successfully!")
    print(f"   - Test Accuracy: {test_acc:.2f}%")
    print(f"   - Test Loss: {test_loss:.4f}")
    print(f"   - Results saved to: {output_dir}/")

    return report_df


def plot_per_class_performance(report_df, class_names, output_dir):
    """
    Create visualization of performance metrics for each class

    Args:
        report_df: DataFrame containing classification report
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Extract class-wise metrics (exclude average rows)
    class_data = report_df.loc[class_names]

    # Create performance comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Precision, Recall, F1-Score by class
    metrics = ['precision', 'recall', 'f1-score']
    x_pos = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax1.bar(x_pos + i * width, class_data[metric], width,
                label=metric.capitalize(), alpha=0.8)

    ax1.set_xlabel('Class Names', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # Plot 2: Support (number of samples) by class
    ax2.bar(class_names, class_data['support'], color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Class Names', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('Class Distribution (Support)', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_training_history(trainer, output_dir):
    """
    Analyze training history for insights into model convergence

    Args:
        trainer: DASTrainer instance with training history
        output_dir: Directory to save analysis results
    """
    if not trainer.train_losses:
        print("   ⚠️  No training history available for analysis")
        return

    # Create convergence analysis plot
    plt.figure(figsize=(12, 8))

    epochs = range(1, len(trainer.train_losses) + 1)

    # Plot training and validation metrics
    plt.subplot(2, 2, 1)
    plt.plot(epochs, trainer.train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, trainer.val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Convergence', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, trainer.train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, trainer.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Progression', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, trainer.learning_rates, 'g-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule', fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    # Calculate generalization gap
    if len(trainer.train_accs) == len(trainer.val_accs):
        generalization_gap = [train - val for train, val in zip(trainer.train_accs, trainer.val_accs)]
        plt.plot(epochs, generalization_gap, 'purple', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Gap (%)')
        plt.title('Generalization Gap (Train - Val)', fontweight='bold')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate and print training insights
    final_train_acc = trainer.train_accs[-1] if trainer.train_accs else 0
    final_val_acc = trainer.val_accs[-1] if trainer.val_accs else 0
    generalization_gap = final_train_acc - final_val_acc

    print("   📊 Training Insights:")
    print(f"      - Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"      - Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"      - Generalization Gap: {generalization_gap:.2f}%")
    print(f"      - Total Epochs Trained: {len(trainer.train_losses)}")


def compare_architectures(results_files):
    """
    Compare performance across different model architectures and strategies

    Args:
        results_files: List of paths to result JSON files from different experiments

    Returns:
        comparison_df: DataFrame containing comparative analysis
    """
    print("📊 Performing Architecture Comparison Analysis...")

    comparisons = []

    # Load results from all experiment files
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)

            # Extract key metrics for comparison
            comparison_data = {
                'model': result.get('model_type', 'Unknown'),
                'strategy': result.get('finetune_strategy', 'N/A'),
                'test_accuracy': result.get('test_accuracy', 0),
                'best_val_accuracy': result.get('best_val_accuracy', 0),
                'test_loss': result.get('test_loss', 0),
                'num_classes': result.get('num_classes', 0),
                'total_parameters': result.get('total_parameters', 0),
                'trainable_parameters': result.get('trainable_parameters', 0)
            }
            comparisons.append(comparison_data)

        except Exception as e:
            print(f"   ⚠️  Could not load results from {file_path}: {e}")

    # Create comparative analysis DataFrame
    comparison_df = pd.DataFrame(comparisons)

    # Print comprehensive comparison
    print("\n🏆 Architecture Performance Comparison:")
    print("=" * 80)
    for idx, comp in enumerate(comparisons):
        print(f"{idx + 1:2d}. {comp['model']:15} | {comp['strategy']:15} | "
              f"Test Acc: {comp['test_accuracy']:6.2f}% | "
              f"Val Acc: {comp['best_val_accuracy']:6.2f}% | "
              f"Params: {comp['total_parameters']:,}")

    # Create visualization for architecture comparison
    create_architecture_comparison_plot(comparisons)

    # Calculate and display rankings
    display_performance_rankings(comparison_df)

    return comparison_df


def create_architecture_comparison_plot(comparisons):
    """
    Create comprehensive visualization comparing different architectures

    Args:
        comparisons: List of comparison dictionaries
    """
    if not comparisons:
        print("   ⚠️  No comparison data available")
        return

    # Create multi-panel comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data for plotting
    model_labels = [f"{comp['model']}\n({comp['strategy']})" for comp in comparisons]
    test_accuracies = [comp['test_accuracy'] for comp in comparisons]
    val_accuracies = [comp['best_val_accuracy'] for comp in comparisons]
    parameters = [comp['total_parameters'] for comp in comparisons]

    # Plot 1: Test accuracy comparison
    bars1 = ax1.bar(range(len(comparisons)), test_accuracies, color='skyblue', alpha=0.7)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_xticks(range(len(comparisons)))
    ax1.set_xticklabels(model_labels, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars1, test_accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Validation vs Test accuracy
    x_pos = np.arange(len(comparisons))
    width = 0.35
    ax2.bar(x_pos - width / 2, val_accuracies, width, label='Validation', alpha=0.7)
    ax2.bar(x_pos + width / 2, test_accuracies, width, label='Test', alpha=0.7)
    ax2.set_title('Validation vs Test Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([comp['model'] for comp in comparisons], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Model size vs performance
    ax3.scatter(parameters, test_accuracies, s=100, alpha=0.7,
                c=val_accuracies, cmap='viridis')
    ax3.set_title('Model Size vs Performance', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Parameters')
    ax3.set_ylabel('Test Accuracy (%)')
    ax3.grid(True, alpha=0.3)

    # Add model labels to scatter points
    for i, (param, acc, label) in enumerate(zip(parameters, test_accuracies,
                                                [comp['model'] for comp in comparisons])):
        ax3.annotate(label, (param, acc), xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.8)

    # Plot 4: Performance efficiency (accuracy per million parameters)
    efficiency = [acc / (param / 1e6) if param > 0 else 0
                  for acc, param in zip(test_accuracies, parameters)]
    ax4.bar(range(len(comparisons)), efficiency, color='lightgreen', alpha=0.7)
    ax4.set_title('Performance Efficiency\n(Accuracy per Million Parameters)',
                  fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy per Million Params')
    ax4.set_xticks(range(len(comparisons)))
    ax4.set_xticklabels([comp['model'] for comp in comparisons], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results/architecture_comparison_comprehensive.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ Architecture comparison visualization saved!")


def display_performance_rankings(comparison_df):
    """
    Display ranked performance of different architectures

    Args:
        comparison_df: DataFrame containing comparison data
    """
    if comparison_df.empty:
        return

    # Rank by test accuracy
    ranked_by_accuracy = comparison_df.sort_values('test_accuracy', ascending=False)

    print("\n🥇 Performance Rankings:")
    print("=" * 60)
    for idx, (_, row) in enumerate(ranked_by_accuracy.iterrows()):
        rank_emoji = ["🥇", "🥈", "🥉"][idx] if idx < 3 else f"{idx + 1:2d}."
        print(f"{rank_emoji} {row['model']:15} ({row['strategy']:15}): "
              f"{row['test_accuracy']:6.2f}% test accuracy")

    # Calculate and display key insights
    best_model = ranked_by_accuracy.iloc[0]
    worst_model = ranked_by_accuracy.iloc[-1]
    accuracy_range = best_model['test_accuracy'] - worst_model['test_accuracy']

    print(f"\n💡 Key Insights:")
    print(f"   - Best performing model: {best_model['model']} ({best_model['strategy']})")
    print(f"   - Performance range: {accuracy_range:.2f}%")
    print(f"   - Number of architectures compared: {len(comparison_df)}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Analysis Module...")

    # Example of how to use these functions
    print("""
    Usage Example:

    1. For ablation study:
       report = perform_ablation_study(trainer, test_loader, class_names)

    2. For architecture comparison:
       results_files = [
           'training_results/das_vit_results.json',
           'training_results/multiscale_vit_results.json',
           'training_results/das_cnn_results.json'
       ]
       comparison = compare_architectures(results_files)
    """)

    print("✅ Analysis module ready for use!")