"""Evaluation script for chord recognition models."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from src.models.chord_models import create_model
from src.features.extractors import FeatureExtractor
from src.data.dataset import ChordDataModule, SyntheticChordDataset
from src.metrics.chord_metrics import ChordEvaluator
from src.utils.device import get_device, set_seed


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Dict:
    """Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load checkpoint on.
        
    Returns:
        Checkpoint dictionary.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    chord_vocab: List[str],
    save_path: str = None,
) -> None:
    """Plot confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        chord_vocab: List of chord vocabulary.
        save_path: Path to save the plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=chord_vocab,
        yticklabels=chord_vocab,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Chord Recognition Confusion Matrix')
    plt.xlabel('Predicted Chord')
    plt.ylabel('True Chord')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_metrics_history(history: Dict, save_path: str = None) -> None:
    """Plot training metrics history.
    
    Args:
        history: Training history dictionary.
        save_path: Path to save the plot.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 score plot
    axes[1, 0].plot(epochs, history['val_f1'], 'm-', label='Validation F1')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(epochs, history['val_accuracy'], 'g-', label='Accuracy')
    axes[1, 1].plot(epochs, history['val_f1'], 'm-', label='F1 Score')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics history plot saved to {save_path}")
    
    plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate chord recognition model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Path to output directory"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset for evaluation"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get device
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, device)
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(
        sample_rate=config["data"]["sample_rate"],
        hop_length=config["data"]["hop_length"],
        n_fft=config["data"]["n_fft"],
        n_mels=config["data"]["n_mels"],
        chroma_method=config["features"]["chroma"]["method"],
        include_mfcc=config["features"].get("mfcc", {}).get("n_mfcc", 0) > 0,
        include_spectral=config["features"].get("spectral", {}).get("spectral_centroid", False),
    )
    
    # Create model
    model = create_model(
        model_type=config["model"]["type"],
        input_size=feature_extractor.get_feature_dimensions()["chroma"],
        num_classes=len(config["evaluation"]["chord_vocab"]),
        **config["model"].get(config["model"]["type"], {})
    )
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {config['model']['type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset
    if args.synthetic or not Path(args.data_dir).exists():
        print("Using synthetic dataset for evaluation")
        dataset = SyntheticChordDataset(
            num_samples=1000,
            feature_extractor=feature_extractor,
            chord_vocab=config["evaluation"]["chord_vocab"],
            segment_length=config["data"]["segment_length"],
            sample_rate=config["data"]["sample_rate"],
        )
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        
    else:
        print(f"Using real dataset from {args.data_dir}")
        data_module = ChordDataModule(
            data_dir=args.data_dir,
            feature_extractor=feature_extractor,
            chord_vocab=config["evaluation"]["chord_vocab"],
            batch_size=config["training"]["batch_size"],
            segment_length=config["data"]["segment_length"],
            overlap=config["data"]["overlap"],
            test_size=0.2,
            val_size=0.2,
            random_state=42,
        )
        
        _, _, test_loader = data_module.get_data_loaders()
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ChordEvaluator(config["evaluation"]["chord_vocab"])
    
    # Get detailed evaluation results
    test_results = evaluator.evaluate_model(
        model, test_loader, device, return_predictions=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(f"F1 Score: {test_results['f1']:.4f}")
    print(f"Precision: {test_results['precision']:.4f}")
    print(f"Recall: {test_results['recall']:.4f}")
    print(f"CSR (Chord Symbol Recall): {test_results['csr']:.4f}")
    print(f"WCSR (Weighted Chord Symbol Recall): {test_results['wcsr']:.4f}")
    print(f"Chord Accuracy: {test_results['chord_accuracy']:.4f}")
    
    # Per-class metrics
    print("\nPer-class F1 Scores:")
    for chord, f1 in test_results['f1_per_class'].items():
        print(f"  {chord}: {f1:.4f}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        
        # Confusion matrix
        confusion_matrix_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(
            test_results['labels'],
            test_results['predictions'],
            config["evaluation"]["chord_vocab"],
            str(confusion_matrix_path)
        )
        
        # Training history if available
        if "training_history" in checkpoint:
            history_path = output_dir / "training_history.png"
            plot_metrics_history(
                checkpoint["training_history"],
                str(history_path)
            )
    
    # Save detailed results
    results = {
        "checkpoint_path": args.checkpoint,
        "config": config,
        "test_metrics": test_results,
        "model_info": {
            "type": config["model"]["type"],
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
        }
    }
    
    results_path = output_dir / "evaluation_results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\nDetailed results saved to {results_path}")
    
    # Generate classification report
    report_path = output_dir / "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("Chord Recognition Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {config['model']['type']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Device: {device}\n\n")
        
        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"  F1 Score: {test_results['f1']:.4f}\n")
        f.write(f"  Precision: {test_results['precision']:.4f}\n")
        f.write(f"  Recall: {test_results['recall']:.4f}\n")
        f.write(f"  CSR: {test_results['csr']:.4f}\n")
        f.write(f"  WCSR: {test_results['wcsr']:.4f}\n\n")
        
        f.write("Per-class F1 Scores:\n")
        for chord, f1 in test_results['f1_per_class'].items():
            f.write(f"  {chord}: {f1:.4f}\n")
    
    print(f"Classification report saved to {report_path}")
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
