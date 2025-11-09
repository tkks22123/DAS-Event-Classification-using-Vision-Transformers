"""
DAS Training Pipeline
=====================
Complete training pipeline for DAS event classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import time
import json
from pathlib import Path


class DASTrainer:
    """
    Complete training pipeline for DAS event classification
    """

    def __init__(self, model, model_name="das_model", device=None):
        self.model = model
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

        # Create output directory
        self.output_dir = Path("training_results")
        self.output_dir.mkdir(exist_ok=True)

        print(f"🚀 Trainer initialized on device: {self.device}")
        print(f"📁 Output directory: {self.output_dir}")

    def setup_training(self, learning_rate=1e-4, weight_decay=1e-4):
        """Setup optimizer and loss function"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )

        self.criterion = nn.CrossEntropyLoss()

        # Move model to device
        self.model.to(self.device)

        print(f"✅ Training setup completed:")
        print(f"   - Optimizer: AdamW (lr={learning_rate})")
        print(f"   - Loss: CrossEntropy")
        print(f"   - Scheduler: CosineAnnealing")

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # Debug: Print input shape and batch progress
            if batch_idx % 10 == 0:
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss tracking...')

            # Ensure input shape is (B, 1, 256, 256)
            if data.dim() == 3:
                data = data.unsqueeze(1)
            elif data.shape[2:] != (256, 256):
                data = data.view(data.size(0), 1, 256, 256)

            self.optimizer.zero_grad()

            outputs = self.model(data)
            loss = self.criterion(outputs, targets)

            loss.backward()

            # Add gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Print actual loss every 10 batches
            if batch_idx % 10 == 0:
                current_loss = loss.item()
                print(f'   Batch {batch_idx}/{len(train_loader)}, Loss: {current_loss:.4f}')

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Ensure input shape is (B, 1, 256, 256)
                if data.dim() == 3:
                    data = data.view(data.size(0), 1, 256, 256)
                elif data.shape[2:] != (256, 256):
                    data = data.view(data.size(0), 1, 256, 256)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=10):
        """Complete training loop"""
        print(f"🎯 Starting training for {epochs} epochs")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")

        best_val_acc = 0.0
        patience_counter = 0
        best_model_state = None

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\n📍 Epoch {epoch + 1}/{epochs}")
            print("-" * 50)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            self.learning_rates.append(current_lr)

            print(f"📈 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"📉 Learning Rate: {current_lr:.2e}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': best_val_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, self.output_dir / f"{self.model_name}_best.pth")

                print(f"💾 New best model saved with val_acc: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ Early stopping counter: {patience_counter}/{early_stopping_patience}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.output_dir / f"{self.model_name}_epoch_{epoch + 1}.pth")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Restored best model with val_acc: {best_val_acc:.2f}%")

        training_time = time.time() - start_time
        print(f"\n⏰ Training completed in {training_time:.2f} seconds")

        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc,
            'training_time': training_time
        }, self.output_dir / f"{self.model_name}_final.pth")

        return best_val_acc

    def evaluate_model(self, test_loader):
        """Comprehensive model evaluation"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Ensure input shape is (B, 1, 256, 256)
                if data.dim() == 3:
                    data = data.view(data.size(0), 1, 256, 256)
                elif data.shape[2:] != (256, 256):
                    data = data.view(data.size(0), 1, 256, 256)

                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)

        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )

        print(f"\n🎯 Model Evaluation Results:")
        print(f"   - Test Loss: {test_loss:.4f}")
        print(f"   - Test Accuracy: {test_acc:.2f}%")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall: {recall:.4f}")
        print(f"   - F1-Score: {f1:.4f}")
        print(f"   - Correct/Total: {correct}/{total}")

        # Print classification report
        print(f"\n📊 Classification Report:")
        print(classification_report(all_targets, all_predictions, zero_division=0))

        return test_acc, test_loss, np.array(all_predictions), np.array(all_targets)

    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("⚠️ No training history to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        epochs = range(1, len(self.train_losses) + 1)

        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracies
        ax2.plot(epochs, self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot learning rate
        ax3.plot(epochs, self.learning_rates, 'g-', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)

        # Plot accuracy gap
        if len(self.train_accs) == len(self.val_accs):
            accuracy_gap = [train - val for train, val in zip(self.train_accs, self.val_accs)]
            ax4.plot(epochs, accuracy_gap, 'purple', linewidth=2)
            ax4.set_title('Training-Validation Accuracy Gap')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Accuracy Gap (%)')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{self.model_name}_training_history.png", dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Training history saved to {self.output_dir}/{self.model_name}_training_history.png")

    def save_training_report(self, class_names, test_accuracy):
        """Save comprehensive training report"""
        report = {
            'model_name': self.model_name,
            'best_validation_accuracy': max(self.val_accs) if self.val_accs else 0,
            'test_accuracy': test_accuracy,
            'final_training_accuracy': self.train_accs[-1] if self.train_accs else 0,
            'final_validation_accuracy': self.val_accs[-1] if self.val_accs else 0,
            'total_epochs_trained': len(self.train_losses),
            'final_learning_rate': self.learning_rates[-1] if self.learning_rates else 0,
            'class_names': class_names,
            'training_losses': self.train_losses,
            'validation_losses': self.val_losses,
            'training_accuracies': self.train_accs,
            'validation_accuracies': self.val_accs,
            'learning_rates': self.learning_rates,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        report_file = self.output_dir / f"{self.model_name}_training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"📄 Training report saved to {report_file}")
