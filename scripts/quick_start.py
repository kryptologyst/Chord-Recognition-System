#!/usr/bin/env python3
"""Quick start script for chord recognition system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🎵 Chord Recognition System - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -e .", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Train a simple model with synthetic data
    print("\n🚀 Training a simple model with synthetic data...")
    if not run_command(
        "python scripts/train.py --synthetic --model_type simple --num_epochs 10 --batch_size 32",
        "Training model"
    ):
        print("❌ Training failed")
        sys.exit(1)
    
    # Evaluate the model
    print("\n📊 Evaluating the trained model...")
    checkpoint_path = Path("outputs/checkpoints/best_model.pt")
    if checkpoint_path.exists():
        if not run_command(
            f"python scripts/evaluate.py --checkpoint {checkpoint_path} --synthetic --plot",
            "Evaluating model"
        ):
            print("⚠️  Evaluation failed, but model was trained successfully")
    else:
        print("⚠️  No checkpoint found, skipping evaluation")
    
    # Start the demo
    print("\n🎯 Starting interactive demo...")
    print("   The demo will open in your web browser")
    print("   Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run(["streamlit", "run", "demo/streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except subprocess.CalledProcessError:
        print("❌ Failed to start demo")
        print("   Make sure Streamlit is installed: pip install streamlit")
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Explore the demo at http://localhost:8501")
    print("2. Check the outputs/ directory for training results")
    print("3. Read the README.md for more detailed usage")
    print("4. Try training with your own data")


if __name__ == "__main__":
    main()
