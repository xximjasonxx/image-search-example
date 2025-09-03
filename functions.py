"""
Helper functions for image analysis using Azure AI Vision.
"""

import logging
import os
from urllib.parse import urlparse
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential


def get_image_analysis_client():
    """Initialize and return Azure AI Vision client."""
    endpoint = os.environ.get("AZURE_AI_VISION_ENDPOINT")
    api_key = os.environ.get("AZURE_AI_VISION_KEY")
    
    if not endpoint or not api_key:
        raise ValueError("AZURE_AI_VISION_ENDPOINT and AZURE_AI_VISION_KEY environment variables must be set")
    
    return ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key)
    )


def extract_blob_url_from_subject(subject: str) -> str:
    """Extract blob URL from EventGrid event subject.
    
    Subject format is typically: /blobServices/default/containers/{container}/blobs/{blob}
    """
    if not subject:
        raise ValueError("Event subject is empty")
    
    # Extract container and blob name from subject
    parts = subject.split('/')
    if len(parts) < 6 or 'containers' not in parts or 'blobs' not in parts:
        raise ValueError(f"Invalid subject format: {subject}")
    
    container_idx = parts.index('containers')
    blob_idx = parts.index('blobs')
    
    if container_idx + 1 >= len(parts) or blob_idx + 1 >= len(parts):
        raise ValueError(f"Cannot extract container/blob from subject: {subject}")
    
    container_name = parts[container_idx + 1]
    blob_name = '/'.join(parts[blob_idx + 1:])  # Handle blob names with slashes
    
    # Get storage account name from environment or construct URL
    storage_account = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    if not storage_account:
        raise ValueError("AZURE_STORAGE_ACCOUNT_NAME environment variable must be set")
    
    blob_url = f"https://{storage_account}.blob.core.windows.net/{container_name}/{blob_name}"
    return blob_url


def vectorize_image(client: ImageAnalysisClient, image_url: str) -> dict:
    """Vectorize image using Azure AI Vision and return analysis results."""
    try:
        # Analyze the image for various visual features
        result = client.analyze_from_url(
            image_url=image_url,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.OBJECTS,
                VisualFeatures.TAGS,
                VisualFeatures.READ  # OCR
            ]
        )
        
        # Extract relevant information for vectorization
        analysis_data = {
            "image_url": image_url,
            "caption": result.caption.text if result.caption else None,
            "confidence": result.caption.confidence if result.caption else None,
            "dense_captions": [
                {"text": caption.text, "confidence": caption.confidence, "bounding_box": caption.bounding_box}
                for caption in (result.dense_captions.list if result.dense_captions else [])
            ],
            "objects": [
                {"name": obj.tags[0].name if obj.tags else "unknown", "confidence": obj.tags[0].confidence if obj.tags else 0, "bounding_box": obj.bounding_box}
                for obj in (result.objects.list if result.objects else [])
            ],
            "tags": [
                {"name": tag.name, "confidence": tag.confidence}
                for tag in (result.tags.list if result.tags else [])
            ],
            "text": [
                {"text": line.text, "bounding_box": line.bounding_polygon}
                for block in (result.read.blocks if result.read else [])
                for line in block.lines
            ]
        }
        
        return analysis_data
        
    except Exception as e:
        logging.error(f"Error analyzing image {image_url}: {str(e)}")
        raise


def is_image_file(url: str) -> bool:
    """Check if the URL points to an image file based on file extension."""
    parsed_url = urlparse(url)
    file_extension = parsed_url.path.lower().split('.')[-1] if '.' in parsed_url.path else ''
    image_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'tiff', 'tif'}
    return file_extension in image_extensions
