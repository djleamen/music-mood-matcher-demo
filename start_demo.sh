#!/bin/bash

echo "🎵 Starting AI Music Mood Matcher Demo..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import streamlit, pandas, numpy, plotly, nltk, textblob, sklearn" 2>/dev/null || {
    echo "📦 Installing missing dependencies..."
    pip3 install -r requirements.txt
}

echo "🚀 Starting demo application..."
echo "🌐 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the application"
echo ""

# Start the demo
streamlit run demo.py
