"""Main training script for chord recognition."""

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from omegaconf import OmegaConf

from src.models.chord_models import create_model, count_parameters
from src.features.extractors import FeatureExtractor
from src.data.dataset import ChordDataModule, SyntheticChordDataset
from src.train.trainer import ChordTrainer
from src.utils.device import get_device, set_seed, get_device_info
from src.metrics.chord_metrics import ChordEvaluator


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_synthetic_dataset(config: Dict) -> SyntheticChordDataset:
    """Create synthetic dataset for demonstration.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Synthetic dataset.
    """
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
    
    # Create synthetic dataset
    dataset = SyntheticChordDataset(
        num_samples=config["data"].get("num_synthetic_samples", 1000),
        feature_extractor=feature_extractor,
        chord_vocab=config["evaluation"]["chord_vocab"],
        segment_length=config["data"]["segment_length"],
        sample_rate=config["data"]["sample_rate"],
    )
    
    return dataset


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train chord recognition model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Path to output directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="crnn",
        choices=["crnn", "transformer", "simple"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.model_type:
        config["model"]["type"] = args.model_type
    if args.num_epochs:
        config["training"]["num_epochs"] = args.num_epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.device if args.device != "auto" else None)
    print(f"Using device: {device}")
    print(f"Device info: {get_device_info()}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Create dataset
    if args.synthetic or not Path(args.data_dir).exists():
        print("Using synthetic dataset")
        dataset = create_synthetic_dataset(config)
        
        # Create data loaders manually for synthetic dataset
        from torch.utils.data import DataLoader, random_split
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=0,  # Synthetic data doesn't need workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
        )
        
    else:
        print(f"Using real dataset from {args.data_dir}")
        # Create data module
        data_module = ChordDataModule(
            data_dir=args.data_dir,
            feature_extractor=feature_extractor,
            chord_vocab=config["evaluation"]["chord_vocab"],
            batch_size=config["training"]["batch_size"],
            segment_length=config["data"]["segment_length"],
            overlap=config["data"]["overlap"],
            test_size=0.2,
            val_size=0.2,
            random_state=args.seed,
        )
        
        train_loader, val_loader, test_loader = data_module.get_data_loaders()
    
    # Create model
    model = create_model(
        model_type=config["model"]["type"],
        input_size=feature_extractor.get_feature_dimensions()["chroma"],
        num_classes=len(config["evaluation"]["chord_vocab"]),
        **config["model"].get(config["model"]["type"], {})
    )
    
    print(f"Model: {config['model']['type']}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = ChordTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        chord_vocab=config["evaluation"]["chord_vocab"],
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        optimizer=config["training"]["optimizer"],
        scheduler=config["training"]["scheduler"],
        gradient_clip_norm=config["training"]["gradient_clip_norm"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        log_dir=str(output_dir / "logs"),
        save_dir=str(output_dir / "checkpoints"),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    training_history = trainer.train(config["training"]["num_epochs"])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    evaluator = ChordEvaluator(config["evaluation"]["chord_vocab"])
    test_metrics = evaluator.evaluate_model(model, test_loader, device)
    
    print("Test Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 Score: {test_metrics['f1']:.4f}")
    print(f"  CSR: {test_metrics['csr']:.4f}")
    print(f"  WCSR: {test_metrics['wcsr']:.4f}")
    
    # Save final results
    results = {
        "config": config,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_info": {
            "type": config["model"]["type"],
            "parameters": count_parameters(model),
            "device": str(device),
        }
    }
    
    results_path = output_dir / "results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"Results saved to {results_path}")
    
    # Close trainer
    trainer.close()


if __name__ == "__main__":
    main()
