"""Training module for chord recognition models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..models.chord_models import create_model, count_parameters
from ..metrics.chord_metrics import ChordEvaluator
from ..utils.device import get_device, move_to_device, clear_memory


class ChordTrainer:
    """Trainer for chord recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        chord_vocab: List[str],
        device: Optional[torch.device] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        scheduler: str = "cosine",
        gradient_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        log_dir: str = "logs",
        save_dir: str = "checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            chord_vocab: List of chord vocabulary.
            device: Device to train on.
            learning_rate: Learning rate.
            weight_decay: Weight decay for regularization.
            optimizer: Optimizer type ('adam', 'adamw', 'sgd').
            scheduler: Learning rate scheduler ('cosine', 'step', 'plateau').
            gradient_clip_norm: Gradient clipping norm.
            early_stopping_patience: Early stopping patience.
            log_dir: Directory for logging.
            save_dir: Directory for saving checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.chord_vocab = chord_vocab
        self.device = device or get_device()
        self.gradient_clip_norm = gradient_clip_norm
        self.early_stopping_patience = early_stopping_patience
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer, learning_rate, weight_decay)
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(scheduler)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize evaluator
        self.evaluator = ChordEvaluator(chord_vocab)
        
        # Initialize logging
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }
    
    def _create_optimizer(
        self,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float,
    ) -> optim.Optimizer:
        """Create optimizer.
        
        Args:
            optimizer_type: Type of optimizer.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            
        Returns:
            Optimizer instance.
        """
        if optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self, scheduler_type: str) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler.
        
        Args:
            scheduler_type: Type of scheduler.
            
        Returns:
            Scheduler instance or None.
        """
        if scheduler_type.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=100,  # Will be updated during training
            )
        elif scheduler_type.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
            )
        elif scheduler_type.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            return None
    
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (features, labels) in enumerate(progress_bar):
            # Move to device
            features = move_to_device(features, self.device)
            labels = move_to_device(labels, self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            
            # Handle different output shapes
            if outputs.dim() == 3:  # (batch_size, seq_len, num_classes)
                outputs = outputs.mean(dim=1)  # Average over sequence
            
            # Compute loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}",
            })
        
        return total_loss / num_batches
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                # Move to device
                features = move_to_device(features, self.device)
                labels = move_to_device(labels, self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Handle different output shapes
                if outputs.dim() == 3:  # (batch_size, seq_len, num_classes)
                    outputs = outputs.mean(dim=1)  # Average over sequence
                
                # Compute loss
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        # Compute validation metrics
        val_metrics = self.evaluator.evaluate_model(
            self.model, self.val_loader, self.device
        )
        
        val_metrics["loss"] = total_loss / num_batches
        
        return val_metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train.
            
        Returns:
            Training history.
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["accuracy"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_loss, val_metrics)
            
            # Update training history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1"])
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}")
            print(f"  Val CSR: {val_metrics['csr']:.4f}")
            print(f"  Val WCSR: {val_metrics['wcsr']:.4f}")
            
            # Early stopping
            if self._should_early_stop(val_metrics["accuracy"]):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save best model
            if val_metrics["accuracy"] > self.best_val_score:
                self.best_val_score = val_metrics["accuracy"]
                self._save_checkpoint(epoch, is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Clear memory
            clear_memory()
        
        # Save final model
        self._save_checkpoint(epoch, is_best=False)
        
        return self.training_history
    
    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
    ) -> None:
        """Log metrics to tensorboard.
        
        Args:
            epoch: Current epoch.
            train_loss: Training loss.
            val_metrics: Validation metrics.
        """
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
        self.writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
        self.writer.add_scalar("F1/Validation", val_metrics["f1"], epoch)
        self.writer.add_scalar("CSR/Validation", val_metrics["csr"], epoch)
        self.writer.add_scalar("WCSR/Validation", val_metrics["wcsr"], epoch)
        
        # Learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("Learning_Rate", current_lr, epoch)
    
    def _should_early_stop(self, val_score: float) -> bool:
        """Check if training should stop early.
        
        Args:
            val_score: Current validation score.
            
        Returns:
            True if training should stop.
        """
        return self.patience_counter >= self.early_stopping_patience
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch.
            is_best: Whether this is the best model.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_score": self.best_val_score,
            "training_history": self.training_history,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_score = checkpoint["best_val_score"]
        self.training_history = checkpoint["training_history"]
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}")
        print(f"Best validation score: {self.best_val_score:.4f}")
    
    def close(self) -> None:
        """Close trainer and cleanup resources."""
        self.writer.close()
        clear_memory()
