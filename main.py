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
import re
import nltk
import logging
import time
import os
import requests
from urllib.parse import urlparse, parse_qs
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__, static_folder=".")
CORS(app)

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

# Function to get video transcript with caching
@lru_cache(maxsize=100)  # Cache up to 100 video transcripts
def get_video_transcript(video_id):
    try:
        start_time = time.time()
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        if not transcript:
            return None, "No transcript available for this video."
        logging.info(f"Transcript fetched in {time.time() - start_time:.2f} seconds")
        return transcript, None
    except TranscriptsDisabled:
        return None, "Transcripts are disabled for this video."
    except NoTranscriptFound:
        return None, "No transcript found for this video."
    except Exception as e:
        return None, str(e)

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

        # Use BART model for summarization
        summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

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
        keywords = [word for word, count in sorted_words[:8]]

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

# Serve the frontend
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

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

# Original API Endpoint (keeping for backward compatibility)
@app.route('/summarize', methods=['GET'])
def summarize_video_get():
    video_url = request.args.get('video_url')
    if not video_url:
        return jsonify({"error": "Missing video_url parameter"}), 400

    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL"}), 400

    # Get transcript
    transcript, error = get_video_transcript(video_id)
    if error:
        return jsonify({"error": error}), 400

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

    # Return JSON response
    return jsonify({
        "summary": summary,
        "keywords": keywords,
        "topics": topics,
        "sentiment": sentiment
    })

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
            return jsonify({"error": error}), 400

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

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({"error": f"Error processing video: {str(e)}"}), 500

# Download NLTK data at module level to ensure it's available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    logging.info("NLTK data is available")
except LookupError:
    logging.warning("NLTK data not found, downloading now...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Create index.html if it doesn't exist
if not os.path.exists('index.html'):
    with open('index.html', 'w') as f:
        f.write("<!-- Frontend HTML will be here -->")
    logging.info("Created placeholder index.html")

if __name__ == "__main__":
    try:
        logging.info("Starting QuikSummarizer API")

        # Get port from environment variable for Render compatibility
        port = int(os.environ.get("PORT", 5000))
        logging.info(f"Server starting on 0.0.0.0:{port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        logging.error(f"Failed to start server: {str(e)}")