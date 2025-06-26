#!/bin/bash
# MANTIS Advanced Miner Setup Script

set -e

echo "🚀 Setting up MANTIS Advanced Miner..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r miner_requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p models
mkdir -p storage

# Create .env template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env template..."
    cat > .env << EOF
# Cloudflare R2 Configuration
R2_ACCOUNT_ID=your_r2_account_id
R2_WRITE_ACCESS_KEY_ID=your_r2_access_key_id
R2_WRITE_SECRET_ACCESS_KEY=your_r2_secret_access_key

# Optional: Custom settings
# MINING_INTERVAL=60
# LOCK_TIME_SECONDS=30
EOF
    echo "⚠️  Please edit .env file with your R2 credentials"
fi

# Set executable permissions
chmod +x advanced_miner.py

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your R2 credentials"
echo "2. Update miner_config.py with your wallet info"
echo "3. Run: python advanced_miner.py --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY"
echo ""
echo "For commit-only mode: python advanced_miner.py --wallet.name YOUR_WALLET --wallet.hotkey YOUR_HOTKEY --commit-only"