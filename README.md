# DAS Event Classification using Vision Transformers

A comprehensive deep learning pipeline for Distributed Acoustic Sensing (DAS) event classification using Vision Transformers (ViT) and multi-scale architectures.

## Project Overview

This project implements state-of-the-art vision transformer architectures for classifying seismic events from DAS data. The system converts 1D acoustic signals into 2D spectrograms and applies transformer-based models for accurate event classification.

## Architecture Features

### Model Variants
* Standard ViT: Single-scale Vision Transformer (ViT-Base configuration)
* Multi-Scale ViT: Processes spectrograms at multiple resolutions (8x8, 16x16, 32x32 patches)
* CNN Baseline: Convolutional neural network for performance comparison

### Multi-Scale Processing
* Fine-scale (8x8): Captures high-frequency details (1024 patches)
* Medium-scale (16x16): Balanced feature extraction (256 patches)
* Coarse-scale (32x32): Global pattern recognition (64 patches)
* Cross-scale Attention: Fuses multi-resolution features

## Project Structure

```
das/
├── main.py # Main training pipeline with analysis
├── das_vit_model.py # Standard ViT and CNN models
├── multi_scale_vit.py # Multi-scale ViT implementation
├── training_pipeline.py # Training utilities and loops
├── data_preprocessor.py # Data loading and preprocessing
├── analysis.py # Comprehensive model analysis tools
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Quick Start

### Installation

```
pip install -r requirements.txt
```

### Basic Usage
#### full finetune
```
python main.py --data_path /path/to/dataset --model_type vit --finetune_strategy full_finetune --learning_rate 1e-6 --epochs 50
```

#### Standard ViT with linear probing
```
python main.py --data_path /path/to/dataset --model_type vit --finetune_strategy linear_probe
```

#### Multi-scale ViT training

```
python main.py --data_path /path/to/dataset --model_type multiscale_vit --epochs 50
```

#### CNN baseline
```
python main.py --data_path /path/to/dataset --model_type cnn --batch_size 32
```

#### Multi-scale ViT with custom configuration
```
python main.py --data_path /path/to/dataset --model_type multiscale_vit --depth_per_scale 6 --learning_rate 1e-4
```

## Configuration Options

### Model Types
* vit: Standard Vision Transformer (ViT-Base)
* multiscale_vit: Multi-scale ViT with feature fusion
* cnn: Convolutional Neural Network baseline

### Fine-tuning Strategies
* full_finetune: Train all layers (default) 
* linear_probe: Freeze backbone, train only classification head
* last_layers: Freeze early layers, train later layers
* differential_lr: Different learning rates for backbone and head

### Key Parameters

```
--data_path PATH # Path to dataset directory (required)
--model_type [vit|multiscale_vit|cnn] # Model architecture
--finetune_strategy STRATEGY # Fine-tuning approach
--epochs N # Training epochs (default: 50)
--batch_size N # Batch size (default: 32)
--learning_rate LR # Learning rate (default: 1e-4)
--max_channels N # Max channels per file (default: 20)
--depth_per_scale N # Transformer depth per scale (multi-scale only)
--use_cache # Use dataset cache (default: True)
```

## Data Processing Pipeline

### Input Data Format
* HDF5 files containing raw DAS acoustic data
* Automatic class labeling based on filename patterns
* Support for multi-channel processing

### Preprocessing Steps
* Signal Segmentation: Split continuous signals into windows (512 samples with 128 overlap)
* Spectrogram Conversion: STFT transformation to 2D representations
* Frequency Filtering: Band-limited processing (50-500 Hz)
* Normalization: Standard scaling and log transformation
* Resizing: Fixed size output (256x256 pixels)

### Supported Event Classes
car, construction, fence, longboard, manipulation, openclose, regular, running, walk

## Model Architectures
### Standard ViT (ViT-Base)
```
DASViT(
img_size=(256, 256),
patch_size=16,
in_channels=1,
num_classes=9,
embed_dim=768,
depth=12, # Total transformer blocks
num_heads=12,
dropout=0.1
)
```

### Multi-Scale ViT
```
MultiScaleViT(
img_size=256,
in_channels=1,
num_classes=9,
embed_dim=768,
depth_per_scale=4, # Transformer blocks per scale
num_heads=12,
dropout=0.1
)
```

### CNN Baseline
```
DASCNN(
in_channels=1,
num_classes=9
)
```

## Training Features
### Advanced Training Utilities
* Cosine Annealing LR: Smooth learning rate scheduling
* Gradient Clipping: Prevents gradient explosion (max_norm=1.0)
* Early Stopping: Automatic training termination (patience=15)
* Model Checkpoints: Best model preservation
* Training Visualization: Loss/accuracy curves
* Attention Visualization: Model interpretability

### Performance Monitoring
* Real-time training progress with batch-level logging
* Validation accuracy tracking
* Test set evaluation with comprehensive metrics
* Automatic analysis and comparison reports

## Analysis Capabilities
### Comprehensive Evaluation
* Ablation Studies: Component contribution analysis
* Confusion Matrices: Error pattern visualization
* Per-Class Performance: Precision, recall, F1-score analysis
* Architecture Comparison: Cross-model performance benchmarking
* Training Convergence: Learning dynamics analysis

### Generated Analysis Files
```
training_results/
├── confusion_matrix.png # Error analysis
├── per_class_performance.png # Class-wise metrics
├── training_convergence_analysis.png # Learning dynamics
├── architecture_comparison.png # Model benchmarking
├── detailed_classification_report.csv # Performance metrics
└── [model]_analysis.json # Comprehensive results
```

## Outputs and Results
### Generated Files
```
training_results/
├── model_best.pth # Best performing model weights
├── model_final.pth # Final model after training
├── training_history.png # Loss and accuracy curves
├── training_report.json # Comprehensive training summary
├── attention_maps.png # ViT attention visualization
└── architecture_comparison.csv # Cross-model performance
```

### Performance Metrics
* Classification accuracy (primary metric)
* Precision, recall, F1-score (per-class and weighted)
* Confusion matrices
* Training/validation convergence curves
* Model efficiency analysis



