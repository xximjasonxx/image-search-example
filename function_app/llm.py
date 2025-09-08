import os
import logging
import requests
from typing import List

def vectorize_text(text: str) -> List[float]:
    """
    Vectorize text using Azure Computer Vision (same model as images).
    
    Args:
        text (str): The text to vectorize
        
    Returns:
        List[float]: Float array representing the text vectorization
    """
    try:
        # Get required environment variables
        endpoint = os.environ.get("AZURE_AI_VISION_ENDPOINT")
        api_key = os.environ.get("AZURE_AI_VISION_KEY")
        
        if not endpoint or not api_key:
            raise ValueError("AZURE_AI_VISION_ENDPOINT and AZURE_AI_VISION_KEY environment variables are required")
        
        # Construct the text vectorization API URL
        vectorize_url = f"{endpoint.rstrip('/')}/computervision/retrieval:vectorizeText"
        
        headers = {
            "Ocp-Apim-Subscription-Key": api_key,
            "Content-Type": "application/json"
        }
        
        params = {
            "api-version": "2024-02-01",
            "model-version": "2023-04-15"
        }
        
        # Request body with text
        data = {
            "text": text
        }
        
        # Make API request
        response = requests.post(
            vectorize_url,
            headers=headers,
            params=params,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")
        
        vector_data = response.json()
        vector = vector_data.get("vector", [])
        
        if not vector:
            raise Exception("No vector returned from API")
        
        logging.info(f"Successfully vectorized text with {len(vector)} dimensions")
        return vector
        
    except Exception as e:
        logging.error(f"Error vectorizing text: {str(e)}")
        raise
