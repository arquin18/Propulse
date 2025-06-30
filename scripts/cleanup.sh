#!/bin/bash

# Exit on error
set -e

echo "🧹 Cleaning up Propulse development environment..."

# Remove Python cache files
echo "🗑️  Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete

# Remove test cache and coverage files
echo "🗑️  Removing test cache and coverage files..."
rm -rf .pytest_cache
rm -rf .coverage
rm -rf htmlcov
rm -rf .tox

# Remove build artifacts
echo "🗑️  Removing build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Remove logs
echo "🗑️  Removing log files..."
rm -rf backend/logs/*
find . -type f -name "*.log" -delete

# Remove temporary files
echo "🗑️  Removing temporary files..."
find . -type f -name "*.tmp" -delete
find . -type f -name "*.bak" -delete
rm -rf temp/

# Clean up Docker (if running)
if command -v docker &> /dev/null; then
    echo "🐳 Cleaning up Docker resources..."
    # Stop and remove containers
    docker ps -a | grep "propulse" | awk '{print $1}' | xargs -r docker stop
    docker ps -a | grep "propulse" | awk '{print $1}' | xargs -r docker rm
    # Remove images
    docker images | grep "propulse" | awk '{print $3}' | xargs -r docker rmi
fi

# Clean Terraform files (if exists)
if [ -d "infra/terraform" ]; then
    echo "🔧 Cleaning Terraform files..."
    rm -rf infra/terraform/.terraform
    rm -f infra/terraform/*.tfstate*
fi

echo "✨ Cleanup complete!" 