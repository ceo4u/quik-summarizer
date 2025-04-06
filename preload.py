"""
Preload script to initialize the transformers model during build time
to avoid timeout during the first request.
"""
import logging
import nltk
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preload_resources():
    """Preload NLTK data and transformers model"""
    logger.info("Preloading NLTK data...")
    try:
        # Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
    
    logger.info("Preloading transformers model...")
    try:
        # Initialize the model once to download and cache it
        _ = pipeline("summarization", model="facebook/bart-base")
        logger.info("Transformers model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading transformers model: {str(e)}")

if __name__ == "__main__":
    preload_resources()
