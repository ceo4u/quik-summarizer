"""
Preload script to download NLTK data during build time
to avoid timeout during the first request.
"""
import logging
import nltk

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preload_resources():
    """Preload NLTK data"""
    logger.info("Preloading NLTK data...")
    try:
        # Download NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")

if __name__ == "__main__":
    preload_resources()
