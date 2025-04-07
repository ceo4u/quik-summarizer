"""
Transcript proxy module to get YouTube transcripts from alternative sources
when the main YouTube API is rate limited.
"""
import logging
import requests
import re
import json
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_transcript_from_alternative_source(video_id):
    """
    Try to get transcript from alternative sources when YouTube API is rate limited.
    Returns (transcript_list, error_message)
    """
    # List of methods to try
    methods = [
        _try_invidious_api,
        _try_youtube_transcript_mirror
    ]
    
    # Try each method
    for method in methods:
        try:
            logger.info(f"Trying alternative transcript source: {method.__name__} for video {video_id}")
            transcript, error = method(video_id)
            if transcript:
                return transcript, None
            # If we got a specific error that indicates no transcript is available,
            # don't try other methods
            if error and ("no transcript" in error.lower() or 
                         "disabled" in error.lower() or
                         "not available" in error.lower()):
                return None, error
        except Exception as e:
            logger.warning(f"Error in {method.__name__}: {str(e)}")
    
    # If all methods failed
    return None, "Could not retrieve transcript from any alternative source"

def _try_invidious_api(video_id):
    """Try to get transcript from an Invidious instance"""
    # List of public Invidious instances
    instances = [
        "https://invidious.snopyta.org",
        "https://yewtu.be",
        "https://invidious.kavin.rocks",
        "https://vid.puffyan.us"
    ]
    
    # Shuffle the list to distribute load
    random.shuffle(instances)
    
    for instance in instances:
        try:
            # Add a small delay to avoid overwhelming the instance
            time.sleep(random.uniform(0.5, 1.5))
            
            # Request the captions
            url = f"{instance}/api/v1/captions/{video_id}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if captions are available
                if not data or len(data) == 0:
                    continue
                
                # Find English captions or use the first available
                caption_track = None
                for track in data:
                    if track.get('languageCode') == 'en':
                        caption_track = track
                        break
                
                if not caption_track and len(data) > 0:
                    caption_track = data[0]
                
                if caption_track:
                    # Get the actual transcript content
                    caption_url = f"{instance}/api/v1/captions/{video_id}?label={caption_track.get('label')}"
                    caption_response = requests.get(caption_url, timeout=5)
                    
                    if caption_response.status_code == 200:
                        # Convert to the format expected by our application
                        lines = []
                        for item in caption_response.json():
                            if 'text' in item:
                                lines.append({
                                    'text': item['text'],
                                    'start': item.get('start', 0),
                                    'duration': item.get('duration', 0)
                                })
                        
                        if lines:
                            return lines, None
            
            # If we get a 404, the video might not have captions
            elif response.status_code == 404:
                return None, "No transcript available for this video"
                
        except Exception as e:
            logger.warning(f"Error with Invidious instance {instance}: {str(e)}")
    
    return None, "Could not retrieve transcript from Invidious"

def _try_youtube_transcript_mirror(video_id):
    """Try to get transcript from a hypothetical YouTube transcript mirror service"""
    # This is a placeholder for a potential future implementation
    # In a real application, you might have your own mirror service or use another API
    return None, "YouTube transcript mirror not implemented"

# Example usage:
# transcript, error = get_transcript_from_alternative_source("video_id_here")
