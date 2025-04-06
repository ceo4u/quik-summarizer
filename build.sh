#!/bin/bash
# Download NLTK data during build phase
python -m nltk.downloader punkt stopwords wordnet

# Run the preload script to initialize the model
python preload.py

# Make the script executable
chmod +x build.sh

echo "Build completed successfully"
