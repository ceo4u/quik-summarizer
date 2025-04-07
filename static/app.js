// Log that the script has loaded
console.log('QuikSummarizer app.js loaded');

// Function to generate summary
function generateSummary() {
    // Get values
    const youtubeUrl = document.getElementById('youtube-url').value;
    const summaryLength = document.getElementById('summary-length').value;

    // Validate URL
    if (!youtubeUrl || !youtubeUrl.includes('youtube.com/watch?v=')) {
        showError("Please enter a valid YouTube URL");
        return;
    }

    // Show loading, hide results and errors
    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display = 'none';
    document.getElementById('error-message').style.display = 'none';

    // Prepare request data
    const data = {
        url: youtubeUrl,
        length: summaryLength
    };

    // Send POST request to backend using relative URL
    // Try the main API endpoint first, with fallback to the original endpoint
    let apiUrl = '/api/summarize';

    // Add a function to retry with fallback endpoint if the main one fails
    const fetchWithFallback = async (url, options) => {
        try {
            console.log(`Attempting to fetch from ${url}`);
            const response = await fetch(url, options);
            console.log(`Response from ${url}:`, response.status);
            return response;
        } catch (error) {
            console.warn(`Failed to fetch from ${url}, trying fallback...`, error);
            // If the main endpoint fails, try the fallback
            console.log('Attempting fallback to /summarize');
            return fetch('/summarize', options);
        }
    };

    fetchWithFallback(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        // Always parse the JSON first
        return response.json().then(data => {
            // If response is not OK, format the error with the parsed data
            if (!response.ok) {
                const error = new Error(data.error || 'An error occurred');
                error.status = response.status;
                error.details = data.details || '';
                error.responseData = data; // Keep the full response data for potential use
                throw error;
            }
            return data;
        });
    })
    .then(data => {
        // Hide loading
        document.getElementById('loading').style.display = 'none';

        // Display results
        document.getElementById('video-title').textContent = data.title;
        document.getElementById('video-channel').textContent = data.channel;
        document.getElementById('video-stats').textContent = data.stats;

        // Set thumbnail
        if (data.thumbnail) {
            document.getElementById('video-thumbnail').style.backgroundImage = `url(${data.thumbnail})`;
        }

        // Set summary
        document.getElementById('summary-text').textContent = data.summary;

        // Set key points
        const keyPointsList = document.getElementById('key-points-list');
        keyPointsList.innerHTML = '';
        data.keyPoints.forEach(point => {
            const li = document.createElement('li');
            li.textContent = point;
            keyPointsList.appendChild(li);
        });

        // Show result section
        document.getElementById('result').style.display = 'block';

        // Log success
        console.log('Summary generated successfully');
    })
    .catch(error => {
        console.error('Error details:', error);
        document.getElementById('loading').style.display = 'none';

        // More detailed error logging
        const errorMessage = error.message || 'Failed to generate summary. Please try again.';
        console.error('Error message:', errorMessage);
        console.error('Error status:', error.status);
        console.error('Error details:', error.details);
        console.error('Request URL:', apiUrl);
        console.error('Request data:', data);

        // Handle specific error types
        if (error.status === 429) {
            // Rate limiting error
            showError('YouTube is temporarily limiting our access. Please try again in a few minutes.');
        } else if (error.status === 404 && error.responseData && error.responseData.thumbnail) {
            // No transcript available but we have video info
            showError('This video doesn't have captions/transcripts available. Try another video.');

            // Show a partial result with the video thumbnail
            document.getElementById('video-thumbnail').style.backgroundImage =
                `url(${error.responseData.thumbnail})`;
            document.getElementById('video-title').textContent =
                error.responseData.title || 'Video without transcript';
            document.getElementById('result').style.display = 'block';
            document.getElementById('summary-text').textContent =
                'No transcript available for this video. Please try a different video.';
        } else {
            // Generic error
            showError(errorMessage);
        }
    });
}

function showError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    console.error('Error shown to user:', message);
}
