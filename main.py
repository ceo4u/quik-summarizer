from flask import Flask, request, jsonify, send_from_directory
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import pipeline
from textblob import TextBlob
from functools import lru_cache
# Import our transcript proxy for alternative sources
import transcript_proxy
import re
import nltk
import logging
import time
import os
import requests
import tempfile
import random
from urllib.parse import urlparse, parse_qs
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set up a custom NLTK data directory in a writable location
# This is necessary for Render deployment
logging.info("Setting up NLTK data...")
try:
    # Create a temporary directory for NLTK data
    nltk_data_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.insert(0, nltk_data_dir)
    logging.info(f"NLTK data directory set to: {nltk_data_dir}")

    # Check if NLTK data is available, download if needed
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        logging.info("NLTK data is available")
    except LookupError:
        logging.warning("NLTK data not found. Downloading now...")
        # Download to the custom directory
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('stopwords', download_dir=nltk_data_dir)
        nltk.download('wordnet', download_dir=nltk_data_dir)
        logging.info("NLTK data downloaded successfully")
except Exception as e:
    logging.error(f"Error with NLTK data: {str(e)}")

# Create index.html if it doesn't exist
if not os.path.exists('index.html'):
    with open('index.html', 'w') as f:
        f.write("<!-- Frontend HTML will be here -->")
    logging.info("Created placeholder index.html")

# Create static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
    logging.info("Created static folder")

# Initialize Flask app with proper static folder configuration
app = Flask(__name__, static_folder="static", static_url_path="/static")

# Enable CORS for all routes and all origins
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "methods": "*"}})

# Add logging for requests
@app.before_request
def log_request_info():
    logging.info(f"Request: {request.method} {request.path} {request.remote_addr}")

# Add after request handler for CORS headers
@app.after_request
def after_request(response):
    # Add CORS headers to all responses
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    video_id = None
    patterns = [
        r'v=([^&]+)',            # For URLs with 'v=' parameter
        r'youtu.be/([^?]+)',     # For shortened URLs
        r'youtube.com/embed/([^?]+)'  # For embed URLs
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break

    # If patterns didn't work, try parsing the URL
    if not video_id:
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc:
            query_params = parse_qs(parsed_url.query)
            if 'v' in query_params:
                video_id = query_params['v'][0]
        elif 'youtu.be' in parsed_url.netloc:
            video_id = parsed_url.path.lstrip('/')

    return video_id

# Improved function to get video info from the YouTube API
def get_video_info(video_id):
    try:
        # For a real implementation, you would use the YouTube Data API
        # This would require adding an API key from the Google Cloud Console
        # For example:
        # api_key = "YOUR_YOUTUBE_API_KEY"
        # url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics,contentDetails&id={video_id}&key={api_key}"
        # response = requests.get(url)
        # data = response.json()

        # For now, we'll extract some info from the video page as a fallback
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(url)

            # Very basic parsing to extract title (not recommended for production)
            title_match = re.search(r'<title>(.*?)</title>', response.text)
            title = title_match.group(1).replace(' - YouTube', '') if title_match else f"Video {video_id}"

            # Extract channel name (basic approach)
            channel_match = re.search(r'"channelName":"(.*?)"', response.text)
            channel = channel_match.group(1) if channel_match else "YouTube Channel"

            return {
                "title": title,
                "channel": channel,
                "stats": "YouTube Video",  # Without API key, we can't get accurate stats
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            }
        except Exception as e:
            logging.error(f"Error fetching video page info: {str(e)}")
            # Return basic info using video ID
            return {
                "title": f"Video {video_id}",
                "channel": "YouTube Channel",
                "stats": "YouTube Video",
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            }

    except Exception as e:
        logging.error(f"Error in video info retrieval: {str(e)}")
        return {
            "title": f"Video {video_id}",
            "channel": "YouTube Channel",
            "stats": "YouTube Video",
            "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
        }

# Function to get video transcript with caching, rate limiting and retries
@lru_cache(maxsize=100)  # Cache up to 100 video transcripts
def get_video_transcript(video_id):
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            start_time = time.time()

            # Add a small random delay to avoid rate limiting
            if attempt > 0:
                delay = retry_delay * (attempt + 1) + (random.random() * 2)
                logging.info(f"Retry attempt {attempt+1}/{max_retries} for video {video_id}. Waiting {delay:.2f} seconds...")
                time.sleep(delay)

            # Try to get the transcript
            transcript = YouTubeTranscriptApi.get_transcript(video_id)

            if not transcript:
                return None, "No transcript available for this video."

            logging.info(f"Transcript fetched in {time.time() - start_time:.2f} seconds")
            return transcript, None

        except TranscriptsDisabled:
            # No need to retry for these specific errors
            return None, "Transcripts are disabled for this video."

        except NoTranscriptFound:
            # No need to retry for these specific errors
            return None, "No transcript found for this video."

        except Exception as e:
            error_message = str(e)
            logging.warning(f"Error fetching transcript (attempt {attempt+1}/{max_retries}): {error_message}")

            # Check if it's a rate limiting error
            if "Too Many Requests" in error_message or "429" in error_message:
                if attempt < max_retries - 1:
                    # Will retry after delay
                    continue
                else:
                    # Try alternative sources when YouTube API is rate limited
                    logging.info(f"YouTube API rate limited. Trying alternative sources for video {video_id}")
                    alt_transcript, alt_error = transcript_proxy.get_transcript_from_alternative_source(video_id)

                    if alt_transcript:
                        logging.info(f"Successfully retrieved transcript from alternative source for video {video_id}")
                        return alt_transcript, None
                    else:
                        # If alternative sources also failed, return the error
                        logging.warning(f"Alternative sources also failed for video {video_id}: {alt_error}")
                        return None, "YouTube is rate limiting requests and alternative sources failed. Please try again in a few minutes."
            else:
                # For other errors, no need to retry
                return None, f"Could not retrieve a transcript for the video. {error_message}"

# Function to split text into chunks
def split_text(text, chunk_size=500):  # Reduced chunk size for safety
    sentences = sent_tokenize(text)
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        current_length += len(sentence) + 1
        if current_length > chunk_size:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = len(sentence) + 1
        current_chunk.append(sentence)
    if current_chunk:
        yield ' '.join(current_chunk)

# Fallback summarization function that doesn't use transformers
def generate_fallback_summary(text, length='medium'):
    try:
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()

        # Split into sentences
        sentences = sent_tokenize(text)

        if not sentences:
            return "No text available to summarize."

        # Calculate sentence scores based on word frequency
        word_frequencies = {}

        # Count word frequencies
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word not in stopwords.words('english'):
                    if word not in word_frequencies:
                        word_frequencies[word] = 1
                    else:
                        word_frequencies[word] += 1

        # Normalize frequencies
        if word_frequencies:
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies:
                word_frequencies[word] = word_frequencies[word] / max_frequency

        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            for word in word_tokenize(sentence.lower()):
                if word in word_frequencies:
                    if i not in sentence_scores:
                        sentence_scores[i] = word_frequencies[word]
                    else:
                        sentence_scores[i] += word_frequencies[word]

        # Determine summary length based on parameter
        if length == 'short':
            summary_length = min(3, len(sentences))
        elif length == 'long':
            summary_length = min(7, len(sentences))
        else:  # medium
            summary_length = min(5, len(sentences))

        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:summary_length]
        top_sentences = sorted(top_sentences, key=lambda x: x[0])  # Sort by position in text

        # Create summary
        summary = ' '.join([sentences[i] for i, _ in top_sentences])

        return summary
    except Exception as e:
        logging.error(f"Error in fallback summarization: {str(e)}")
        # Last resort fallback
        sentences = sent_tokenize(text)
        if sentences and len(sentences) >= 3:
            return sentences[0] + ' ' + sentences[1] + ' ' + sentences[2]
        elif sentences:
            return ' '.join(sentences[:min(len(sentences), 3)])
        else:
            return "Unable to generate summary."

# Improved function to summarize text
def summarize_text(text, length='medium'):
    try:
        # Clean the text to remove problematic patterns
        # Remove repeated advertisement text
        text = re.sub(r'advertisement+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Limit input length for summarization model
        if len(text) > 10000:
            text = text[:10000]

        # Use smaller chunks for better processing
        text_chunks = list(split_text(text, chunk_size=1000))

        # Set summary parameters based on length preference
        if length == 'short':
            max_length = 75
            min_length = 30
        elif length == 'long':
            max_length = 300
            min_length = 150
        else:  # medium is default
            max_length = 150
            min_length = 75

        # Create a new pipeline instance each time (slower but more reliable)
        try:
            # Use a smaller, faster model that's more likely to work on Render
            summarization_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
            logging.info("Loaded summarization model successfully")
        except Exception as e:
            logging.error(f"Failed to load summarization model: {str(e)}")
            # Try a fallback approach - extract key sentences
            logging.warning("Using fallback summarization method")
            return generate_fallback_summary(text, length)

        summaries = []
        for chunk in text_chunks[:3]:  # Process up to 3 chunks to avoid excessive processing
            try:
                if not chunk or len(chunk) < 50:
                    continue

                summary = summarization_pipeline(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )

                if summary and len(summary) > 0:
                    summary_text = summary[0]['summary_text']
                    # Clean up the summary text
                    summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                    summaries.append(summary_text)
            except Exception as e:
                logging.error(f"Error summarizing chunk: {str(e)}")
                # Simple fallback
                sentences = sent_tokenize(chunk)
                if sentences and len(sentences) > 1:
                    summaries.append(sentences[0])

        if not summaries:
            return "Unable to generate a summary for this video."

        # Join the summaries with proper spacing
        final_summary = ' '.join(summaries)

        # Add periods if missing at the end of sentences
        final_summary = re.sub(r'([a-zA-Z])\s+([A-Z])', r'\1. \2', final_summary)

        return final_summary

    except Exception as e:
        logging.error(f"Summarization error: {str(e)}")
        return "An error occurred during summarization."

# Improved function to extract keywords
def extract_keywords(text):
    try:
        # Clean the text
        text = re.sub(r'advertisement+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Use NLTK for better keyword extraction
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        # Split into sentences and select a subset for processing
        sentences = sent_tokenize(text)
        processed_text = ' '.join(sentences[:20])  # Use first 20 sentences

        # Tokenize and normalize words
        words = word_tokenize(processed_text.lower())
        words = [lemmatizer.lemmatize(word) for word in words
                if word.isalnum() and len(word) > 2 and word not in stop_words]

        # Count word frequencies
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

        # Get the most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_words[:8]]

        # Filter out common words that aren't useful as keywords
        common_words = {'video', 'going', 'like', 'just', 'get', 'know', 'make', 'really', 'thing', 'way', 'time'}
        keywords = [word for word in keywords if word not in common_words]

        return keywords[:5]  # Return top 5 filtered keywords

    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        return ["Unable to extract keywords"]

# New function to generate meaningful key points
def generate_key_points(text, keywords):
    try:
        # Use TextBlob for sentiment analysis
        analysis = TextBlob(text)
        sentiment = analysis.sentiment

        # Start with an empty list of key points
        key_points = []

        # Add points based on the top keywords
        if keywords and len(keywords) >= 3:
            for keyword in keywords[:3]:
                # Find sentences containing this keyword
                sentences = [s for s in sent_tokenize(text.lower()) if keyword in s.lower()]
                if sentences:
                    # Use the first sentence containing this keyword
                    point = sentences[0].strip()
                    # Clean up the point
                    point = re.sub(r'^\W+', '', point)  # Remove leading non-word chars
                    point = point[0].upper() + point[1:]  # Capitalize first letter
                    if len(point) > 100:
                        point = point[:97] + "..."
                    key_points.append(point)
                else:
                    key_points.append(f"The video discusses {keyword}")

        # Add sentiment-based point
        if sentiment.polarity > 0.2:
            key_points.append("The content presents a positive perspective on the topic")
        elif sentiment.polarity < -0.2:
            key_points.append("The content presents critical or cautionary viewpoints")
        else:
            key_points.append("The content maintains a balanced perspective")

        # Add style-based point
        if sentiment.subjectivity > 0.6:
            key_points.append("The video focuses on opinions and subjective analysis")
        elif sentiment.subjectivity < 0.3:
            key_points.append("The video emphasizes factual information and objective data")
        else:
            key_points.append("The video balances factual information with analysis")

        # Ensure we have at least 5 points
        if len(key_points) < 5:
            key_points.append("The video provides detailed explanations about the topic")

        return key_points
    except Exception as e:
        logging.error(f"Error generating key points: {str(e)}")
        return [
            "The video discusses the main topic in detail",
            "Several important concepts are explained",
            "The content provides useful information",
            "Multiple aspects of the topic are covered",
            "The video offers insights on the subject matter"
        ]

# Function to perform topic modeling (LDA)
def topic_modeling(text):
    try:
        # Ensure text is long enough for topic modeling
        if len(text.split()) < 20:
            return [["Text too short for topic modeling"]]

        vectorizer = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
        tf = vectorizer.fit_transform([text])

        # Check if we have enough features
        if tf.shape[1] < 5:
            return [["Not enough unique terms for topic modeling"]]

        lda_model = LatentDirichletAllocation(n_components=5, max_iter=5,
                                             learning_method='online', random_state=42)
        lda_model.fit(tf)
        feature_names = vectorizer.get_feature_names_out()

        topics = []
        # Use _ to indicate unused variable
        for _, topic in enumerate(lda_model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[:-6:-1]])
        return topics
    except Exception as e:
        logging.error(f"Error in topic modeling: {str(e)}")
        return [["Error in topic modeling"]]

# Health check endpoint
@app.route('/health')
def health_check():
    response = jsonify({
        "status": "ok",
        "message": "Service is running"
    })
    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    return response

# Serve the frontend
@app.route('/')
def index():
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logging.error(f"Error serving index.html: {str(e)}")
        # Return a simple HTML page as fallback
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>QuikSummarizer</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: Arial; text-align: center; padding: 50px; }
                h1 { color: #9c27b0; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>QuikSummarizer</h1>
            <p>There was an error loading the application.</p>
            <p class="error">Please try refreshing the page or contact support.</p>
        </body>
        </html>
        '''

# Route to serve static files directly
@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory('static', filename)
    except Exception as e:
        logging.error(f"Error serving static file {filename}: {str(e)}")
        return "File not found", 404

# For serving placeholder images
@app.route('/api/placeholder/<width>/<height>')
def placeholder(width, height):
    # Return a simple SVG placeholder
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect width="100%" height="100%" fill="#333"/>
        <text x="50%" y="50%" font-family="Arial" font-size="14" fill="white" text-anchor="middle" dominant-baseline="middle">
            Placeholder {width}x{height}
        </text>
    </svg>'''

    return svg, 200, {'Content-Type': 'image/svg+xml'}

# Handle OPTIONS requests for CORS preflight for /summarize
@app.route('/summarize', methods=['OPTIONS'])
def handle_summarize_options():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

# Original API Endpoint (keeping for backward compatibility)
@app.route('/summarize', methods=['GET', 'POST'])
def summarize_video_get():
    # Handle both GET and POST requests
    if request.method == 'POST':
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        video_url = data.get('url')
    else:  # GET request
        video_url = request.args.get('video_url')

    if not video_url:
        return jsonify({"error": "Missing video URL parameter"}), 400

    # Log the request for debugging
    logging.info(f"Processing request for video URL: {video_url}")

    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    # Get transcript
    transcript, error = get_video_transcript(video_id)
    if error:
        # Check if we should provide a more user-friendly error message
        if "rate limiting" in error.lower():
            return jsonify({
                "error": "YouTube is temporarily limiting our access. Please try again in a few minutes.",
                "details": error
            }), 429  # Use proper rate limit status code
        elif "no transcript" in error.lower() or "transcripts are disabled" in error.lower():
            # For videos without transcripts, return a specific error
            return jsonify({
                "error": "This video doesn't have captions/transcripts available.",
                "details": error,
                "title": f"Video {video_id}",
                "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
            }), 404
        else:
            # Generic error
            return jsonify({
                "error": "Could not process this video.",
                "details": error
            }), 400

    # Combine transcript text
    try:
        video_text = ' '.join([line['text'] for line in transcript])

        # Clean the text to remove problematic characters
        video_text = re.sub(r'[^\w\s.,?!-]', '', video_text)

        # Limit the total text length to avoid memory issues
        if len(video_text) > 10000:  # Limit to 10K characters
            video_text = video_text[:10000]

        # Summarize text
        summary = summarize_text(video_text)
    except Exception as e:
        return jsonify({"error": f"Error processing transcript: {str(e)}"}), 500

    # Extract keywords with error handling
    try:
        keywords = extract_keywords(video_text)
    except Exception as e:
        logging.error(f"Keyword extraction failed: {str(e)}")
        keywords = ["Keyword extraction failed"]

    # Perform topic modeling with error handling
    try:
        topics = topic_modeling(video_text)
    except Exception as e:
        logging.error(f"Topic modeling failed: {str(e)}")
        topics = [["Topic modeling failed"]]

    # Perform sentiment analysis with error handling
    try:
        sentiment_analysis = TextBlob(video_text).sentiment
        sentiment = {
            "polarity": round(sentiment_analysis.polarity, 2),
            "subjectivity": round(sentiment_analysis.subjectivity, 2)
        }
    except Exception as e:
        logging.error(f"Sentiment analysis failed: {str(e)}")
        sentiment = {"polarity": 0, "subjectivity": 0}

    # Log successful processing
    logging.info(f"Successfully processed video ID: {video_id}")

    # Create JSON response
    response = jsonify({
        "summary": summary,
        "keywords": keywords,
        "topics": topics,
        "sentiment": sentiment,
        # Add these fields for compatibility with the frontend
        "title": f"Video {video_id}",
        "channel": "YouTube Channel",
        "stats": "YouTube Video",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg",
        "keyPoints": ["Key point 1", "Key point 2", "Key point 3", "Key point 4", "Key point 5"]
    })

    # Add CORS headers
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    return response

# Handle OPTIONS requests for CORS preflight for /api/summarize
@app.route('/api/summarize', methods=['OPTIONS'])
def handle_api_options():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    return response

# Updated API endpoint for frontend integration
@app.route('/api/summarize', methods=['POST'])
def summarize_video_api():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    video_url = data.get('url')
    summary_length = data.get('length', 'medium')

    if not video_url:
        return jsonify({"error": "Missing video URL"}), 400

    try:
        # Extract video ID
        video_id = extract_video_id(video_url)
        if not video_id:
            return jsonify({"error": "Invalid YouTube URL"}), 400

        # Get video info
        video_info = get_video_info(video_id)

        # Get transcript
        transcript, error = get_video_transcript(video_id)
        if error:
            # Check if we should provide a more user-friendly error message
            if "rate limiting" in error.lower():
                error_response = jsonify({
                    "error": "YouTube is temporarily limiting our access. Please try again in a few minutes.",
                    "details": error
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
                return error_response, 429  # Use proper rate limit status code
            elif "no transcript" in error.lower() or "transcripts are disabled" in error.lower():
                # For videos without transcripts, return a specific error
                error_response = jsonify({
                    "error": "This video doesn't have captions/transcripts available.",
                    "details": error,
                    "title": f"Video {video_id}",
                    "thumbnail": f"https://img.youtube.com/vi/{video_id}/mqdefault.jpg"
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
                return error_response, 404
            else:
                # Generic error
                error_response = jsonify({
                    "error": "Could not process this video.",
                    "details": error
                })
                error_response.headers.add('Access-Control-Allow-Origin', '*')
                error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
                return error_response, 400

        # Process transcript
        video_text = ' '.join([line['text'] for line in transcript])

        # Clean the text
        video_text = re.sub(r'[^\w\s.,?!-]', '', video_text)
        video_text = re.sub(r'\s+', ' ', video_text).strip()

        # Extract keywords (do this before truncating text)
        keywords = extract_keywords(video_text)

        # Limit text length for processing
        if len(video_text) > 15000:
            video_text = video_text[:15000]

        # Generate summary with specified length
        summary = summarize_text(video_text, summary_length)

        # Generate meaningful key points
        key_points = generate_key_points(video_text, keywords)

        # Return comprehensive result
        result = {
            "title": video_info["title"],
            "channel": video_info["channel"],
            "stats": video_info["stats"],
            "thumbnail": video_info["thumbnail"],
            "summary": summary,
            "keyPoints": key_points,
            "keywords": keywords  # Include keywords as additional data
        }

        # Log successful processing
        logging.info(f"Successfully processed video ID: {video_id}")

        # Create response with CORS headers
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')

        return response

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        error_response = jsonify({"error": f"Error processing video: {str(e)}"})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        error_response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        error_response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return error_response, 500

if __name__ == "__main__":
    try:
        logging.info("Starting QuikSummarizer API")

        # Get port from environment variable for Render compatibility
        port = int(os.environ.get("PORT", 5000))
        logging.info(f"Server starting on 0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")
