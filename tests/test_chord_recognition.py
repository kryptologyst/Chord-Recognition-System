"""Tests for chord recognition system."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.chord_models import create_model, count_parameters
from src.features.extractors import FeatureExtractor, ChromaExtractor
from src.data.dataset import SyntheticChordDataset
from src.metrics.chord_metrics import ChordMetrics
from src.utils.device import get_device, set_seed


class TestChordModels:
    """Test chord recognition models."""
    
    def test_create_model(self):
        """Test model creation."""
        model = create_model("simple", input_size=12, num_classes=25)
        assert isinstance(model, torch.nn.Module)
        assert count_parameters(model) > 0
    
    def test_crnn_model(self):
        """Test CRNN model."""
        model = create_model("crnn", input_size=12, num_classes=25)
        
        # Test forward pass
        x = torch.randn(2, 10, 12)  # batch_size=2, seq_len=10, input_size=12
        output = model(x)
        
        assert output.shape == (2, 10, 25)  # batch_size, seq_len, num_classes
    
    def test_transformer_model(self):
        """Test Transformer model."""
        model = create_model("transformer", input_size=12, num_classes=25)
        
        # Test forward pass
        x = torch.randn(2, 10, 12)  # batch_size=2, seq_len=10, input_size=12
        output = model(x)
        
        assert output.shape == (2, 10, 25)  # batch_size, seq_len, num_classes
    
    def test_simple_model(self):
        """Test simple model."""
        model = create_model("simple", input_size=12, num_classes=25)
        
        # Test forward pass with 2D input
        x = torch.randn(2, 12)  # batch_size=2, input_size=12
        output = model(x)
        
        assert output.shape == (2, 25)  # batch_size, num_classes
        
        # Test forward pass with 3D input
        x = torch.randn(2, 10, 12)  # batch_size=2, seq_len=10, input_size=12
        output = model(x)
        
        assert output.shape == (2, 10, 25)  # batch_size, seq_len, num_classes


class TestFeatureExtractors:
    """Test feature extractors."""
    
    def test_chroma_extractor(self):
        """Test chroma feature extraction."""
        extractor = ChromaExtractor(sample_rate=22050)
        
        # Generate synthetic audio
        duration = 3.0
        sr = 22050
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * 440 * t)  # A4 note
        
        # Extract features
        chroma = extractor.extract(audio)
        chroma_mean = extractor.extract_mean(audio)
        
        assert chroma.shape[0] == 12  # 12 pitch classes
        assert chroma_mean.shape == (12,)
        assert np.allclose(np.sum(chroma_mean), 1.0, atol=0.1)  # Should sum to ~1
    
    def test_feature_extractor(self):
        """Test main feature extractor."""
        extractor = FeatureExtractor(
            sample_rate=22050,
            include_mfcc=True,
            include_spectral=True
        )
        
        # Generate synthetic audio
        duration = 3.0
        sr = 22050
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * 440 * t)  # A4 note
        
        # Extract features
        features = extractor.extract_features(audio)
        mean_features = extractor.extract_mean_features(audio)
        
        assert "chroma" in features
        assert "mfcc" in features
        assert "spectral_centroid" in features
        
        assert "chroma" in mean_features
        assert "mfcc" in mean_features
        assert "spectral_centroid" in mean_features


class TestDataset:
    """Test dataset classes."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset."""
        chord_vocab = ["C", "Cm", "D", "Dm", "N"]
        
        extractor = FeatureExtractor(sample_rate=22050)
        dataset = SyntheticChordDataset(
            num_samples=100,
            feature_extractor=extractor,
            chord_vocab=chord_vocab,
            segment_length=3.0,
            sample_rate=22050
        )
        
        assert len(dataset) == 100
        
        # Test getting an item
        features, label = dataset[0]
        assert isinstance(features, torch.Tensor)
        assert features.shape == (12,)  # Chroma features
        assert 0 <= label < len(chord_vocab)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_chord_metrics(self):
        """Test chord metrics computation."""
        chord_vocab = ["C", "Cm", "D", "Dm", "N"]
        metrics = ChordMetrics(chord_vocab)
        
        # Generate test data
        y_true = [0, 1, 2, 3, 4] * 20  # 100 samples
        y_pred = [0, 1, 2, 3, 4] * 20  # Perfect predictions
        
        results = metrics.compute_metrics(y_true, y_pred)
        
        assert "accuracy" in results
        assert "f1" in results
        assert "csr" in results
        assert "wcsr" in results
        
        assert results["accuracy"] == 1.0
        assert results["f1"] == 1.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        chord_vocab = ["C", "Cm", "D", "Dm", "N"]
        metrics = ChordMetrics(chord_vocab)
        
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        
        cm = metrics.compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (5, 5)  # 5x5 confusion matrix


class TestUtils:
    """Test utility functions."""
    
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_seed_setting(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        np_rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()
        
        set_seed(42)
        
        # Generate the same random numbers
        np_rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()
        
        assert np_rand1 == np_rand2
        assert torch_rand1 == torch_rand2


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        set_seed(42)
        
        # Create model
        model = create_model("simple", input_size=12, num_classes=25)
        model.eval()
        
        # Create feature extractor
        extractor = FeatureExtractor(sample_rate=22050)
        
        # Generate synthetic audio
        duration = 3.0
        sr = 22050
        t = np.linspace(0, duration, int(duration * sr))
        audio = np.sin(2 * np.pi * 440 * t)  # A4 note
        
        # Extract features
        features = extractor.extract_features(audio)
        chroma = features["chroma"]
        
        # Convert to tensor
        chroma_tensor = torch.tensor(chroma.mean(axis=1), dtype=torch.float32).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(chroma_tensor)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        assert 0 <= predicted_class < 25


if __name__ == "__main__":
    pytest.main([__file__])
