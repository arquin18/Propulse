#!/bin/bash

# Exit on error
set -e

echo "🚀 Setting up Propulse development environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment from environment.yml
echo "📦 Creating conda environment..."
conda env create -f environment.yml

# Activate environment
echo "🔄 Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate propulse

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p backend/logs
mkdir -p frontend/assets
mkdir -p shared/sample_rfps
mkdir -p shared/templates

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.sample .env
fi

echo "✅ Setup complete! You can now activate the environment with:"
echo "   conda activate propulse" 