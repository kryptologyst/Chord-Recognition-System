"""Evaluation metrics for chord recognition."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class ChordMetrics:
    """Metrics for chord recognition evaluation."""
    
    def __init__(self, chord_vocab: List[str]):
        """Initialize chord metrics.
        
        Args:
            chord_vocab: List of chord vocabulary.
        """
        self.chord_vocab = chord_vocab
        self.chord_to_idx = {chord: idx for idx, chord in enumerate(chord_vocab)}
        self.idx_to_chord = {idx: chord for chord, idx in self.chord_to_idx.items()}
    
    def compute_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        average: str = "macro",
    ) -> Dict[str, float]:
        """Compute comprehensive metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            average: Averaging method for multi-class metrics.
            
        Returns:
            Dictionary of computed metrics.
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # Basic classification metrics
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics["precision_per_class"] = {
            self.idx_to_chord[i]: float(precision_per_class[i])
            for i in range(len(self.chord_vocab))
        }
        metrics["recall_per_class"] = {
            self.idx_to_chord[i]: float(recall_per_class[i])
            for i in range(len(self.chord_vocab))
        }
        metrics["f1_per_class"] = {
            self.idx_to_chord[i]: float(f1_per_class[i])
            for i in range(len(self.chord_vocab))
        }
        
        # Chord-specific metrics
        metrics.update(self._compute_chord_specific_metrics(y_true, y_pred))
        
        return metrics
    
    def _compute_chord_specific_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Compute chord-specific metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Dictionary of chord-specific metrics.
        """
        metrics = {}
        
        # Chord Symbol Recall (CSR)
        metrics["csr"] = self._compute_csr(y_true, y_pred)
        
        # Weighted Chord Symbol Recall (WCSR)
        metrics["wcsr"] = self._compute_wcsr(y_true, y_pred)
        
        # Chord accuracy (ignoring "No chord" class)
        no_chord_idx = self.chord_to_idx.get("N", -1)
        if no_chord_idx >= 0:
            mask = y_true != no_chord_idx
            if np.sum(mask) > 0:
                metrics["chord_accuracy"] = accuracy_score(y_true[mask], y_pred[mask])
            else:
                metrics["chord_accuracy"] = 0.0
        else:
            metrics["chord_accuracy"] = metrics["accuracy"]
        
        return metrics
    
    def _compute_csr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Chord Symbol Recall (CSR).
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            CSR score.
        """
        # CSR is the same as recall for chord recognition
        return recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    def _compute_wcsr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Weighted Chord Symbol Recall (WCSR).
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            WCSR score.
        """
        # Compute class weights based on frequency
        class_counts = np.bincount(y_true)
        total_samples = len(y_true)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights[class_counts == 0] = 0  # Handle zero counts
        
        # Compute weighted recall
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        weighted_recall = np.sum(class_weights * recall_per_class) / np.sum(class_weights)
        
        return float(weighted_recall)
    
    def compute_confusion_matrix(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
    ) -> np.ndarray:
        """Compute confusion matrix.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Confusion matrix.
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
    ) -> str:
        """Get detailed classification report.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            
        Returns:
            Classification report string.
        """
        # Convert to numpy arrays
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        
        return classification_report(
            y_true, y_pred, target_names=self.chord_vocab, zero_division=0
        )


class ChordEvaluator:
    """Evaluator for chord recognition models."""
    
    def __init__(self, chord_vocab: List[str]):
        """Initialize evaluator.
        
        Args:
            chord_vocab: List of chord vocabulary.
        """
        self.chord_vocab = chord_vocab
        self.metrics = ChordMetrics(chord_vocab)
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, List]]:
        """Evaluate model on a dataset.
        
        Args:
            model: PyTorch model.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of evaluation results.
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Handle different output shapes
                if outputs.dim() == 3:  # (batch_size, seq_len, num_classes)
                    outputs = outputs.mean(dim=1)  # Average over sequence
                
                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        results = self.metrics.compute_metrics(all_labels, all_predictions)
        
        if return_predictions:
            results["predictions"] = all_predictions
            results["labels"] = all_labels
        
        return results
    
    def evaluate_sequence_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, List]]:
        """Evaluate sequence model on a dataset.
        
        Args:
            model: PyTorch model.
            data_loader: Data loader for evaluation.
            device: Device to run evaluation on.
            return_predictions: Whether to return predictions.
            
        Returns:
            Dictionary of evaluation results.
        """
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                
                # Get predictions for each time step
                predictions = torch.argmax(outputs, dim=-1)
                
                # Flatten for evaluation
                predictions_flat = predictions.view(-1)
                labels_flat = labels.view(-1)
                
                all_predictions.extend(predictions_flat.cpu().numpy())
                all_labels.extend(labels_flat.cpu().numpy())
        
        # Compute metrics
        results = self.metrics.compute_metrics(all_labels, all_predictions)
        
        if return_predictions:
            results["predictions"] = all_predictions
            results["labels"] = all_labels
        
        return results


def compute_chord_similarity(chord1: str, chord2: str) -> float:
    """Compute similarity between two chords.
    
    Args:
        chord1: First chord.
        chord2: Second chord.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if chord1 == chord2:
        return 1.0
    
    # Simple similarity based on root note
    if chord1 == "N" or chord2 == "N":
        return 0.0
    
    # Extract root notes
    root1 = chord1[0] if chord1 else ""
    root2 = chord2[0] if chord2 else ""
    
    if root1 == root2:
        return 0.5  # Same root, different quality
    
    return 0.0  # Different roots


def compute_chord_transition_accuracy(
    y_true: List[str],
    y_pred: List[str],
) -> float:
    """Compute chord transition accuracy.
    
    Args:
        y_true: True chord sequence.
        y_pred: Predicted chord sequence.
        
    Returns:
        Transition accuracy.
    """
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0
    
    correct_transitions = 0
    total_transitions = len(y_true) - 1
    
    for i in range(total_transitions):
        true_transition = (y_true[i], y_true[i + 1])
        pred_transition = (y_pred[i], y_pred[i + 1])
        
        if true_transition == pred_transition:
            correct_transitions += 1
    
    return correct_transitions / total_transitions
