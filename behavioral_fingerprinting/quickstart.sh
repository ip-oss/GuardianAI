#!/bin/bash

# Quick start script for LLM Behavioral Fingerprinting
# This script helps you set up and run your first experiment

echo "=================================================="
echo "LLM Behavioral Fingerprinting - Quick Start"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  ANTHROPIC_API_KEY is not set"
    echo ""
    echo "Please set your API key:"
    echo "  export ANTHROPIC_API_KEY='your-key-here'"
    echo ""
    echo "Get your API key from: https://console.anthropic.com/"
    echo ""
    read -p "Press Enter after setting the API key, or Ctrl+C to exit..."

    # Check again
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "❌ API key still not set. Exiting."
        exit 1
    fi
fi

echo "✓ ANTHROPIC_API_KEY is set"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✓ Dependencies installed"
echo ""

# Run tests
echo "Running setup tests..."
python3 test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup tests failed. Please check the errors above."
    exit 1
fi

echo ""
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Quick experiment (3 trajectories, ~$3-5, 5-10 min):"
echo "   python3 experiments/exp7_llm_taxonomy.py 3"
echo ""
echo "2. Full experiment (20 trajectories, ~$50-150, 1-2 hours):"
echo "   python3 experiments/exp7_llm_taxonomy.py 20"
echo ""
echo "3. Visualize results:"
echo "   python3 analysis/visualize_llm_results.py"
echo ""
