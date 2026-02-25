"""Interactive demo for chord recognition using Streamlit."""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import torch
import soundfile as sf

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.chord_models import create_model
from src.features.extractors import FeatureExtractor
from src.utils.device import get_device, set_seed
from src.metrics.chord_metrics import ChordMetrics


# Page configuration
st.set_page_config(
    page_title="Chord Recognition System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🎵 Chord Recognition System</h1>', unsafe_allow_html=True)

# Privacy disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Privacy & Research Disclaimer</h4>
    <p><strong>This is a research demonstration system for educational purposes only.</strong></p>
    <ul>
        <li>No audio data is stored or transmitted to external servers</li>
        <li>All processing is performed locally in your browser</li>
        <li>This system is not intended for biometric identification or voice cloning</li>
        <li>Misuse of this technology for deceptive purposes is prohibited</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["simple", "crnn", "transformer"],
    index=0,
    help="Select the neural network architecture"
)

# Chord vocabulary
CHORD_VOCAB = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
    "N"  # No chord
]

# Feature extraction parameters
st.sidebar.subheader("Feature Extraction")
sample_rate = st.sidebar.slider("Sample Rate", 8000, 48000, 22050)
hop_length = st.sidebar.slider("Hop Length", 256, 1024, 512)
n_fft = st.sidebar.slider("FFT Size", 1024, 4096, 2048)
chroma_method = st.sidebar.selectbox("Chroma Method", ["stft", "cqt", "cqt_harmonic"])

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "feature_extractor" not in st.session_state:
    st.session_state.feature_extractor = None
if "device" not in st.session_state:
    st.session_state.device = None

# Load model function
@st.cache_resource
def load_model(model_type: str, device: torch.device):
    """Load the chord recognition model."""
    try:
        # Create feature extractor
        feature_extractor = FeatureExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            chroma_method=chroma_method,
            include_mfcc=False,
            include_spectral=False,
        )
        
        # Create model
        model = create_model(
            model_type=model_type,
            input_size=12,  # Chroma features
            num_classes=len(CHORD_VOCAB),
        )
        
        # Load model weights (if available)
        model_path = Path("checkpoints/best_model.pt")
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            st.success("Loaded pre-trained model")
        else:
            st.warning("No pre-trained model found. Using random weights.")
        
        model.eval()
        return model, feature_extractor
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Initialize model
if st.session_state.model is None:
    device = get_device()
    st.session_state.device = device
    st.session_state.model, st.session_state.feature_extractor = load_model(model_type, device)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎤 Audio Input")
    
    # Audio input options
    input_method = st.radio(
        "Choose input method:",
        ["Upload Audio File", "Record Audio", "Generate Synthetic Chord"]
    )
    
    audio_data = None
    audio_sr = None
    
    if input_method == "Upload Audio File":
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Upload a WAV, MP3, FLAC, or M4A file"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Load audio
                audio_data, audio_sr = librosa.load(tmp_path, sr=sample_rate)
                st.success(f"Loaded audio: {uploaded_file.name}")
                st.info(f"Duration: {len(audio_data) / audio_sr:.2f} seconds")
                
                # Clean up
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading audio: {e}")
    
    elif input_method == "Record Audio":
        st.info("Audio recording functionality would be implemented here using streamlit-audio-recorder")
        # For now, show a placeholder
        if st.button("Generate Sample Audio"):
            # Generate a simple chord
            duration = 3.0
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Generate C major chord
            frequencies = [261.63, 329.63, 392.00]  # C, E, G
            audio_data = np.zeros_like(t)
            for freq in frequencies:
                audio_data += 0.3 * np.sin(2 * np.pi * freq * t)
            
            audio_data += np.random.normal(0, 0.05, len(audio_data))
            audio_sr = sample_rate
            st.success("Generated C major chord sample")
    
    elif input_method == "Generate Synthetic Chord":
        chord_type = st.selectbox("Select chord type:", CHORD_VOCAB[:-1])  # Exclude "N"
        duration = st.slider("Duration (seconds)", 1.0, 10.0, 3.0)
        
        if st.button("Generate Chord"):
            # Generate synthetic chord
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Chord tone mapping
            chord_tones = {
                "C": [261.63, 329.63, 392.00],      # C major
                "C#": [277.18, 349.23, 415.30],     # C# major
                "D": [293.66, 369.99, 440.00],      # D major
                "D#": [311.13, 392.00, 466.16],     # D# major
                "E": [329.63, 415.30, 493.88],     # E major
                "F": [349.23, 440.00, 523.25],      # F major
                "F#": [369.99, 466.16, 554.37],    # F# major
                "G": [392.00, 493.88, 587.33],     # G major
                "G#": [415.30, 523.25, 622.25],    # G# major
                "A": [440.00, 554.37, 659.25],     # A major
                "A#": [466.16, 587.33, 698.46],    # A# major
                "B": [493.88, 622.25, 739.99],     # B major
                "Cm": [261.63, 311.13, 392.00],    # C minor
                "C#m": [277.18, 329.63, 415.30],   # C# minor
                "Dm": [293.66, 349.23, 440.00],    # D minor
                "D#m": [311.13, 369.99, 466.16],   # D# minor
                "Em": [329.63, 392.00, 493.88],    # E minor
                "Fm": [349.23, 415.30, 523.25],    # F minor
                "F#m": [369.99, 440.00, 554.37],   # F# minor
                "Gm": [392.00, 466.16, 587.33],    # G minor
                "G#m": [415.30, 493.88, 622.25],   # G# minor
                "Am": [440.00, 523.25, 659.25],    # A minor
                "A#m": [466.16, 554.37, 698.46],  # A# minor
                "Bm": [493.88, 587.33, 739.99],    # B minor
            }
            
            frequencies = chord_tones.get(chord_type, [261.63, 329.63, 392.00])
            audio_data = np.zeros_like(t)
            for freq in frequencies:
                audio_data += 0.3 * np.sin(2 * np.pi * freq * t)
            
            audio_data += np.random.normal(0, 0.05, len(audio_data))
            audio_sr = sample_rate
            st.success(f"Generated {chord_type} chord")

with col2:
    st.header("🎯 Chord Recognition")
    
    if audio_data is not None:
        # Display audio waveform
        st.subheader("Audio Waveform")
        fig_wave = go.Figure()
        time_axis = np.linspace(0, len(audio_data) / audio_sr, len(audio_data))
        fig_wave.add_trace(go.Scatter(x=time_axis, y=audio_data, mode='lines', name='Waveform'))
        fig_wave.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude",
            height=300
        )
        st.plotly_chart(fig_wave, use_container_width=True)
        
        # Extract features and predict
        if st.session_state.model is not None and st.session_state.feature_extractor is not None:
            try:
                # Extract features
                features = st.session_state.feature_extractor.extract_features(audio_data)
                chroma_features = features["chroma"]
                
                # Display chromagram
                st.subheader("Chromagram")
                fig_chroma = go.Figure(data=go.Heatmap(
                    z=chroma_features,
                    x=list(range(chroma_features.shape[1])),
                    y=[f"C{i}" for i in range(12)],
                    colorscale='Viridis'
                ))
                fig_chroma.update_layout(
                    title="Chromagram",
                    xaxis_title="Time Frames",
                    yaxis_title="Pitch Class",
                    height=300
                )
                st.plotly_chart(fig_chroma, use_container_width=True)
                
                # Predict chord
                with st.spinner("Analyzing chord..."):
                    # Convert to tensor
                    chroma_tensor = torch.tensor(chroma_features.mean(axis=1), dtype=torch.float32).unsqueeze(0)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = st.session_state.model(chroma_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0, predicted_class].item()
                
                # Display results
                predicted_chord = CHORD_VOCAB[predicted_class]
                
                st.subheader("Recognition Results")
                
                # Main prediction
                col_pred, col_conf = st.columns([2, 1])
                with col_pred:
                    st.metric("Predicted Chord", predicted_chord)
                with col_conf:
                    st.metric("Confidence", f"{confidence:.2%}")
                
                # Top predictions
                st.subheader("Top 5 Predictions")
                top_k = 5
                top_probs, top_indices = torch.topk(probabilities[0], top_k)
                
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    chord_name = CHORD_VOCAB[idx.item()]
                    prob_value = prob.item()
                    
                    # Create progress bar
                    st.write(f"{i+1}. {chord_name}: {prob_value:.2%}")
                    st.progress(prob_value)
                
                # Confidence distribution
                st.subheader("Confidence Distribution")
                fig_conf = go.Figure(data=go.Bar(
                    x=CHORD_VOCAB,
                    y=probabilities[0].numpy(),
                    marker_color=['red' if i == predicted_class else 'lightblue' for i in range(len(CHORD_VOCAB))]
                ))
                fig_conf.update_layout(
                    title="Confidence Scores",
                    xaxis_title="Chord",
                    yaxis_title="Probability",
                    height=400
                )
                st.plotly_chart(fig_conf, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        
        else:
            st.error("Model not loaded. Please check the configuration.")
    
    else:
        st.info("Please upload an audio file, record audio, or generate a synthetic chord to see recognition results.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Chord Recognition System - Research Demonstration</p>
    <p>This system is for educational and research purposes only.</p>
</div>
""", unsafe_allow_html=True)
