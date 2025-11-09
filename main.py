import torch
import os
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import argparse
import sys
import json
from pathlib import Path
import urllib.request

# Import custom modules
from data_preprocessor import DASDataProcessor, DASPyTorchDataset
from training_pipeline import DASTrainer
from das_vit_model import DASViT, DASCNN
from multi_scale_vit import MultiScaleViT
from analysis import perform_ablation_study, compare_architectures

# Set random seed for reproducibility
torch.manual_seed(114514)


def setup_environment():
    """Initialize environment and check dependencies"""
    print("🔧 Setting up environment...")

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def parse_arguments():
    """Parse command line arguments for training configuration"""
    parser = argparse.ArgumentParser(description='DAS ViT Training with Multi-Scale Support')

    # Required arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset directory')

    # Model configuration
    parser.add_argument('--model_type', type=str, default='vit',
                        choices=['vit', 'cnn', 'multiscale_vit'],  # Added multiscale_vit
                        help='Model type: vit, cnn, or multiscale_vit')

    # Fine-tuning strategies
    parser.add_argument('--finetune_strategy', type=str, default='full_finetune',
                        choices=['full_finetune', 'linear_probe', 'last_layers', 'differential_lr'],
                        help='Fine-tuning strategy for ViT models')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    # Data processing parameters
    parser.add_argument('--max_channels', type=int, default=20, help='Max channels per file')
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum total samples')
    parser.add_argument('--use_cache', action='store_true', default=True, help='Use dataset cache')

    # Multi-scale specific parameters
    parser.add_argument('--depth_per_scale', type=int, default=4,
                        help='Transformer depth for each scale in multi-scale ViT')

    return parser.parse_args()


def setup_finetune_strategy(model, strategy):
    """Setup fine-tuning strategy by freezing/unfreezing layers"""
    print(f"🎯 Setting up {strategy} fine-tuning strategy...")

    if strategy == 'linear_probe':
        # Freeze all layers except the classification head and patch embedding
        for name, param in model.named_parameters():
            if 'head' not in name and 'patch_embed' not in name:
                param.requires_grad = False
        print("   - Frozen: All backbone layers")
        print("   - Trainable: Patch embedding + Classification head")

    elif strategy == 'last_layers':
        # Freeze first N layers, unfreeze last M layers
        if hasattr(model, 'blocks'):  # Standard ViT
            total_layers = len(model.blocks)
            freeze_layers = total_layers // 2  # Freeze first half

            # Freeze patch embedding and first few layers
            for param in model.patch_embed.parameters():
                param.requires_grad = False

            for i, block in enumerate(model.blocks):
                for param in block.parameters():
                    param.requires_grad = (i >= freeze_layers)

            # Always train the head
            for param in model.head.parameters():
                param.requires_grad = True

            print(f"   - Frozen: Patch embed + First {freeze_layers} transformer layers")
            print(f"   - Trainable: Last {total_layers - freeze_layers} layers + head")

        elif hasattr(model, 'scale_transformers'):  # Multi-scale ViT
            print("   - Multi-scale ViT: Using full fine-tuning for scale-specific transformers")

    elif strategy == 'differential_lr':
        # Different learning rates for different parts
        # This will be handled in the optimizer setup
        print("   - Differential learning rates will be applied in optimizer")

    else:  # full_finetune
        # Train all layers
        for param in model.parameters():
            param.requires_grad = True
        print("   - Full fine-tuning: All layers trainable")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   - Trainable parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.1f}%)")


def load_and_prepare_data(data_path, args):
    """
    Load DAS dataset and prepare for training
    Returns:
        spectrograms: List of spectrogram data
        labels: List of corresponding labels
        class_names: List of class names
        processor: Data processor instance
    """
    print("\n📊 Loading dataset...")
    print(f"   Data path: {data_path}")

    # Check if data path exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path '{data_path}' does not exist!")

    # Initialize data processor
    processor = DASDataProcessor(
        max_channels_per_file=args.max_channels
    )

    # Load data - data preprocessor automatically classifies based on filenames
    spectrograms, labels, class_names = processor.load_dataset_from_folders(
        data_path, use_cache=args.use_cache
    )

    print(f"✅ Data loaded: {len(spectrograms)} samples, {len(class_names)} classes")
    print(f"   Classes: {class_names}")

    return spectrograms, labels, class_names, processor


def create_data_loaders(spectrograms, labels, processor, batch_size=32):
    """Create train/val/test data loaders from preprocessed data"""
    # Create dataset
    dataset = DASPyTorchDataset(spectrograms, labels, processor)

    # Calculate split sizes (70% train, 15% validation, 15% test)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    print(f"\n📈 Dataset splitting:")
    print(f"   - Total samples: {total_size}")
    print(f"   - Training set: {train_size} ({100 * train_size / total_size:.1f}%)")
    print(f"   - Validation set: {val_size} ({100 * val_size / total_size:.1f}%)")
    print(f"   - Test set: {test_size} ({100 * test_size / total_size:.1f}%)")

    # Split dataset with fixed random seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"\n📁 Data loaders created:")
    print(f"   - Training batches: {len(train_loader)}")
    print(f"   - Validation batches: {len(val_loader)}")
    print(f"   - Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def initialize_model(model_type, num_classes, device, args):
    """
    Initialize the DAS model with configuration options
    Args:
        model_type: Type of model ('vit', 'cnn', 'multiscale_vit')
        num_classes: Number of output classes
        device: Target device (CPU/GPU)
        args: Command line arguments
    Returns:
        model: Initialized model
        model_name: Name identifier for the model
    """
    print(f"\n🧠 Initializing {model_type.upper()} model...")
    print(f"   Fine-tune strategy: {args.finetune_strategy}")

    if model_type == "multiscale_vit":
        # Multi-Scale ViT with different patch sizes
        model = MultiScaleViT(
            img_size=256,
            in_channels=1,
            num_classes=num_classes,
            embed_dim=768,
            depth_per_scale=args.depth_per_scale,
            num_heads=12,
            dropout=0.1
        )
        model_name = "multiscale_vit"
        print(f"   - Multi-scale ViT with depth {args.depth_per_scale} per scale")

    elif model_type == "vit":
        # Standard single-scale ViT
        model = DASViT(
            img_size=(256, 256),
            patch_size=16,
            in_channels=1,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            dropout=0.1
        )
        model_name = "das_vit"
        # Setup fine-tuning strategy for standard ViT
        setup_finetune_strategy(model, args.finetune_strategy)

    else:
        # CNN baseline model
        model = DASCNN(
            in_channels=1,
            num_classes=num_classes
        )
        model_name = "das_cnn"
        print("   - CNN model: Training from scratch")

    # Move model to target device
    model.to(device)

    # Print model summary
    total_params, trainable_params = model.get_trainable_parameters()

    print(f"✅ Model initialized successfully:")
    print(f"   - Model type: {model_type.upper()}")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Number of classes: {num_classes}")
    print(f"   - Trainable ratio: {100 * trainable_params / total_params:.1f}%")

    return model, model_name


def create_optimizer(model, learning_rate, strategy):
    """Create optimizer with differential learning rates if needed"""
    if strategy == 'differential_lr':
        # Different learning rates for different parts
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': learning_rate / 10},  # Lower LR for backbone
            {'params': head_params, 'lr': learning_rate}  # Higher LR for head
        ], weight_decay=0.01)

        print(f"   - Differential LR: backbone={learning_rate/10:.2e}, head={learning_rate:.2e}")

    else:
        # Standard optimizer with uniform learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        print(f"   - Uniform LR: {learning_rate:.2e}")

    return optimizer


def train_model(model, train_loader, val_loader, device, model_name, args):
    """
    Train the DAS model with specified configuration
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Target device
        model_name: Name identifier for the model
        args: Training arguments
    Returns:
        trainer: Trained trainer instance
    """
    print("\n⚡ Starting model training...")
    print(f"   Target: >85% validation accuracy")
    print(f"   Model: {model_name}")
    print(f"   Strategy: {args.finetune_strategy}")
    print(f"   Epochs: {args.epochs}")

    # Initialize trainer
    trainer = DASTrainer(model, model_name=model_name, device=device)

    # Setup training configuration
    trainer.setup_training(learning_rate=args.learning_rate)

    # If custom optimizer needed for differential LR
    if args.finetune_strategy == 'differential_lr':
        trainer.optimizer = create_optimizer(model, args.learning_rate, args.finetune_strategy)

    # Start training
    trained_model = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        early_stopping_patience=15
    )

    return trainer


def evaluate_model(trainer, test_loader, class_names, model_name, strategy):
    """
    Evaluate the trained model on test set
    Args:
        trainer: Trained trainer instance
        test_loader: Test data loader
        class_names: List of class names
        model_name: Name identifier for the model
        strategy: Fine-tuning strategy used
    Returns:
        test_accuracy: Test set accuracy
    """
    print("\n🧪 Evaluating model on test set...")
    print(f"   Fine-tune strategy: {strategy}")

    # Evaluate model performance
    test_acc, test_loss, predictions, targets = trainer.evaluate_model(test_loader)

    # Generate comprehensive performance report
    report = {
        'model_type': model_name,
        'finetune_strategy': strategy,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'num_classes': len(class_names),
        'classes': class_names,
        'best_val_accuracy': max(trainer.val_accs) if trainer.val_accs else 0
    }

    # Save results to JSON file
    results_file = f"training_results/{model_name}_{strategy}_results.json"
    with open(results_file, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"💾 Results saved to: {results_file}")

    return test_acc


def perform_comprehensive_analysis(trainer, test_loader, class_names, model_name, strategy):
    """
    Perform comprehensive model analysis including ablation studies
    """
    print("\n🔬 Performing Comprehensive Model Analysis...")

    # Perform ablation study and detailed analysis
    ablation_report = perform_ablation_study(trainer, test_loader, class_names)

    # Save analysis results
    analysis_results = {
        'model_name': model_name,
        'finetune_strategy': strategy,
        'ablation_report': ablation_report.to_dict(),
        'test_accuracy': ablation_report.loc['accuracy', 'f1-score'] * 100,
        'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save analysis results
    analysis_file = f"training_results/{model_name}_{strategy}_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"💾 Comprehensive analysis saved to: {analysis_file}")

    return ablation_report


def run_architecture_comparison():
    """
    Compare performance across different architectures if multiple results exist
    """
    print("\n📊 Running Architecture Comparison...")

    # Find all result files
    results_dir = Path("training_results")
    result_files = list(results_dir.glob("*_results.json"))

    if len(result_files) >= 2:
        print(f"   Found {len(result_files)} result files for comparison")
        comparison_df = compare_architectures(result_files)

        # Save comparison results
        comparison_df.to_csv(results_dir / "architecture_comparison.csv", index=False)
        print("💾 Architecture comparison saved to: training_results/architecture_comparison.csv")
    else:
        print("   ⚠️  Need at least 2 result files for comparison")

    return len(result_files) >= 2


def visualize_results(trainer, test_loader, model_name, strategy):
    """
    Generate visualizations for model interpretation and analysis
    Args:
        trainer: Trained trainer instance
        test_loader: Test data loader
        model_name: Name identifier for the model
        strategy: Fine-tuning strategy used
    """
    print("\n📈 Generating visualizations...")

    # Plot training curves (loss and accuracy)
    print("   - Generating training curves...")
    trainer.plot_training_history()

    # Try to visualize attention if it's a ViT model
    if "vit" in model_name.lower():
        print("   - Generating attention maps...")
        try:
            # Get a sample for attention visualization
            sample_batch, sample_labels = next(iter(test_loader))
            sample_input = sample_batch[0:1]  # Take first sample

            # Ensure correct shape for ViT
            if sample_input.dim() == 3:
                sample_input = sample_input.unsqueeze(1)
            if sample_input.shape[2:] != (256, 256):
                sample_input = sample_input.view(sample_input.size(0), 1, 256, 256)

            # Generate attention visualization
            attention_maps = trainer.visualize_attention(sample_input)
        except Exception as e:
            print(f"   ⚠️  Attention visualization failed: {e}")

    print("✅ Visualizations saved to 'training_results/' directory")


def main():
    """
    Main execution function for DAS training pipeline
    Supports multi-scale ViT, standard ViT, and CNN models
    """
    print("=" * 60)
    print("🎯 DAS Vision Transformer Training Pipeline")
    print("      with Multi-Scale Support")
    print("=" * 60)

    # Parse command line arguments
    args = parse_arguments()

    try:
        # Step 1: Environment setup
        device = setup_environment()

        # Step 2: Load and prepare data
        spectrograms, labels, class_names, processor = load_and_prepare_data(args.data_path, args)

        if len(spectrograms) == 0:
            print("❌ No data loaded. Please check your data path and file structure.")
            return

        # Step 3: Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            spectrograms, labels, processor, args.batch_size
        )

        # Step 4: Initialize model with specified configuration
        model, model_name = initialize_model(args.model_type, len(class_names), device, args)

        # Step 5: Train model with specified strategy
        trainer = train_model(model, train_loader, val_loader, device, model_name, args)


        # Step 6: Perform comprehensive analysis (UPDATED)
        print("\n🔬 Step 6: Comprehensive Model Analysis")
        ablation_report = perform_comprehensive_analysis(
            trainer, test_loader, class_names, model_name, args.finetune_strategy
        )

        # Step 7: Evaluate model on test set
        test_accuracy = evaluate_model(trainer, test_loader, class_names, model_name, args.finetune_strategy)

        # Step 8: Run architecture comparison if multiple experiments exist
        has_comparison = run_architecture_comparison()

        # Step 9: Generate visualizations
        visualize_results(trainer, test_loader, model_name, args.finetune_strategy)


        print("\n" + "=" * 60)
        print("🎉 Training Pipeline Completed Successfully!")
        print("=" * 60)
        print(f"📊 Model: {model_name}")
        print(f"🎯 Strategy: {args.finetune_strategy}")
        print(f"📁 Results saved in: training_results/")
        print(f"📈 Best model: training_results/{model_name}_best.pth")
        print(f"🏆 Test Accuracy: {test_accuracy:.2f}%")

        # Final status assessment
        if test_accuracy >= 85.0:
            print("\n✅ SUCCESS: Target accuracy of 85% achieved!")
        else:
            print(f"\n⚠️  Target not achieved: {test_accuracy:.2f}% < 85%")

    except Exception as e:
        print(f"\n❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("   - Data path correctness")
        print("   - File permissions")
        print("   - Dataset structure")
        sys.exit(1)


if __name__ == "__main__":
    main()