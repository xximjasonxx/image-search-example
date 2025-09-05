import os
import logging
from openai import AzureOpenAI
from typing import List

# Get required environment variables
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not api_key:
    raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")
if not azure_endpoint:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=api_key,
    api_version="2024-02-01",
    azure_endpoint=azure_endpoint
)

def vectorize_text(text: str) -> List[float]:
    """
    Vectorize text using Azure OpenAI embeddings.
    
    Args:
        text (str): The text to vectorize
        
    Returns:
        List[float]: Float array representing the text vectorization
    """
    try:
        # Get the deployment name from environment variables
        deployment_name = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        
        if not deployment_name:
            raise ValueError("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable is not set")
        
        # Create embeddings using Azure OpenAI
        response = client.embeddings.create(
            input=text,
            model=deployment_name
        )
        
        # Extract the embedding vector
        embedding = response.data[0].embedding
        
        logging.info(f"Successfully vectorized text with {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        logging.error(f"Error vectorizing text: {str(e)}")
        raise
