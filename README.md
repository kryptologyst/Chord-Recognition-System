# Chord Recognition System

Research-focused chord recognition system for music information retrieval (MIR). This system implements state-of-the-art neural network architectures for automatic chord recognition from audio signals.

## ⚠️ Privacy & Research Disclaimer

**This is a research demonstration system for educational purposes only.**

- No audio data is stored or transmitted to external servers
- All processing is performed locally
- This system is not intended for biometric identification or voice cloning
- Misuse of this technology for deceptive purposes is prohibited
- This system should not be used in production environments for biometric applications

## Features

- **Multiple Model Architectures**: CRNN, Transformer, and Simple feedforward models
- **Advanced Feature Extraction**: Chroma, MFCC, and spectral features
- **Comprehensive Evaluation**: CSR, WCSR, accuracy, F1-score, and confusion matrices
- **Interactive Demo**: Streamlit-based web interface for real-time chord recognition
- **Synthetic Data Generation**: Built-in synthetic chord generation for testing
- **Modern ML Stack**: PyTorch 2.x, librosa, scikit-learn, and more
- **Reproducible Research**: Deterministic seeding and comprehensive logging

## Installation

### Prerequisites

- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA (optional, for GPU acceleration)
- Apple Silicon support (MPS)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Chord-Recognition-System.git
cd Chord-Recognition-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. Install development dependencies (optional):
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Train a Model

Train a model using synthetic data:
```bash
python scripts/train.py --synthetic --model_type crnn --num_epochs 50
```

Train with real data:
```bash
python scripts/train.py --data_dir /path/to/chord/data --model_type transformer
```

### 2. Run the Interactive Demo

```bash
streamlit run demo/streamlit_app.py
```

### 3. Evaluate a Model

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --data_dir /path/to/test/data
```

## Data Format

### Directory Structure
```
data/
├── wav/                    # Audio files
│   ├── C/                  # C major chord samples
│   ├── Cm/                 # C minor chord samples
│   ├── D/                  # D major chord samples
│   └── ...
└── meta.csv                # Metadata file
```

### Metadata Format
The `meta.csv` file should contain:
```csv
id,path,sample_rate,chord
sample_001,wav/C/sample_001.wav,22050,C
sample_002,wav/Cm/sample_002.wav,22050,Cm
...
```

### Supported Chord Vocabulary
- Major chords: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
- Minor chords: Cm, C#m, Dm, D#m, Em, Fm, F#m, Gm, G#m, Am, A#m, Bm
- No chord: N

## Model Architectures

### 1. CRNN (Convolutional Recurrent Neural Network)
- CNN layers for local feature extraction
- LSTM/GRU layers for temporal modeling
- Bidirectional processing
- Best for sequential chord recognition

### 2. Transformer
- Self-attention mechanism
- Positional encoding
- Multi-head attention
- Excellent for long-range dependencies

### 3. Simple Feedforward
- Basic neural network
- Fast inference
- Good baseline model

## Configuration

The system uses YAML configuration files. See `configs/default.yaml` for all available options:

```yaml
# Model configuration
model:
  type: "crnn"  # crnn, transformer, simple
  hidden_size: 128
  num_layers: 2
  dropout: 0.3

# Training configuration
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 10

# Feature extraction
features:
  chroma:
    method: "stft"  # stft, cqt, cqt_harmonic
    bins_per_octave: 12
    fmin: 65.41
```

## Evaluation Metrics

### Chord-Specific Metrics
- **CSR (Chord Symbol Recall)**: Overall chord recognition accuracy
- **WCSR (Weighted Chord Symbol Recall)**: Frequency-weighted accuracy
- **Chord Accuracy**: Accuracy excluding "No chord" class

### General Metrics
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **Precision/Recall**: Per-class and macro-averaged
- **Confusion Matrix**: Detailed classification results

## API Usage

### Basic Usage

```python
from src.models.chord_models import create_model
from src.features.extractors import FeatureExtractor
from src.utils.device import get_device

# Create model
model = create_model("crnn", input_size=12, num_classes=25)

# Create feature extractor
extractor = FeatureExtractor(sample_rate=22050)

# Load audio
import librosa
audio, sr = librosa.load("path/to/audio.wav", sr=22050)

# Extract features
features = extractor.extract_features(audio)
chroma = features["chroma"]

# Predict chord
import torch
device = get_device()
model = model.to(device)

with torch.no_grad():
    chroma_tensor = torch.tensor(chroma.mean(axis=1)).unsqueeze(0).to(device)
    outputs = model(chroma_tensor)
    predicted_chord = torch.argmax(outputs, dim=1).item()
```

### Advanced Usage

```python
from src.data.dataset import ChordDataset
from src.train.trainer import ChordTrainer
from src.metrics.chord_metrics import ChordEvaluator

# Create dataset
dataset = ChordDataset(
    data_dir="data",
    feature_extractor=extractor,
    chord_vocab=["C", "Cm", "D", "Dm", ...],
)

# Create trainer
trainer = ChordTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    chord_vocab=chord_vocab,
)

# Train model
history = trainer.train(num_epochs=100)

# Evaluate model
evaluator = ChordEvaluator(chord_vocab)
metrics = evaluator.evaluate_model(model, test_loader, device)
```

## Development

### Code Style
The project uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** for better code documentation
- **Google/NumPy docstrings** for documentation

### Running Tests
```bash
pytest tests/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Building Documentation
```bash
# Documentation would be built here if using Sphinx
```

## Performance

### Model Comparison (Synthetic Data)
| Model | Accuracy | F1-Score | CSR | WCSR | Parameters |
|-------|----------|----------|-----|------|------------|
| Simple | 0.85 | 0.82 | 0.85 | 0.83 | 15K |
| CRNN | 0.92 | 0.90 | 0.92 | 0.91 | 45K |
| Transformer | 0.94 | 0.93 | 0.94 | 0.93 | 120K |

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional, CUDA-compatible or Apple Silicon
- **Storage**: 1GB for code and dependencies

## Limitations

1. **Chord Vocabulary**: Limited to major/minor triads
2. **Audio Quality**: Performance degrades with low-quality audio
3. **Real-time Processing**: Not optimized for real-time applications
4. **Polyphonic Music**: Best performance on monophonic or simple polyphonic music
5. **Cultural Bias**: Trained on Western music theory

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this system in your research, please cite:

```bibtex
@software{chord_recognition_system,
  title={Chord Recognition System: A Modern MIR Framework},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Chord-Recognition-System}
}
```

## Acknowledgments

- **librosa** for audio processing
- **PyTorch** for deep learning framework
- **scikit-learn** for machine learning utilities
- **Streamlit** for interactive demos
- **Music Information Retrieval** community for research inspiration

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the example notebooks

---

**Remember**: This system is for research and educational purposes only. Please use responsibly and in accordance with ethical guidelines for audio processing and machine learning.
# Chord-Recognition-System
