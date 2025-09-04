"""
Azure Cognitive Search functions for uploading and managing image data.
"""

import logging
import os
import uuid
from typing import List
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Namespace UUID for image documents (you can use any UUID or generate a custom one)
IMAGE_NAMESPACE = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')  # Standard DNS namespace UUID


def generate_document_id(image_name: str) -> str:
    """Generate a UUID5 based on the image name."""
    return str(uuid.uuid5(IMAGE_NAMESPACE, image_name))


def get_search_client():
    """Initialize and return Azure Cognitive Search client."""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    admin_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")
    
    if not endpoint or not admin_key or not index_name:
        raise ValueError("AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_ADMIN_KEY, and AZURE_SEARCH_INDEX_NAME environment variables must be set")
    
    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(admin_key)
    )


def get_search_index_client():
    """Initialize and return Azure Cognitive Search Index client."""
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    admin_key = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
    
    if not endpoint or not admin_key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_ADMIN_KEY environment variables must be set")
    
    return SearchIndexClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(admin_key)
    )


def upload(image_name: str, url: str, vector: List[float]) -> dict:
    """
    Upload image data to Azure Cognitive Search index.
    
    Args:
        image_name: Name/identifier for the image
        url: URL to the image
        vector: Vector embedding of the image
        
    Returns:
        dict: Upload result with success status and document key
    """
    try:
        # Get search client
        search_client = get_search_client()
        
        # Create document for upload
        document = {
            "id": generate_document_id(image_name),  # Use UUID5 based on image_name
            "filename": image_name,
            "url": url,
            "vector_content": vector
        }
        
        # Upload document to search index
        result = search_client.upload_documents([document])
        
        # Check if upload was successful
        if result and len(result) > 0:
            upload_result = result[0]
            if upload_result.succeeded:
                logging.info(f"Successfully uploaded image '{image_name}' to search index")
                return {
                    "success": True,
                    "document_key": upload_result.key,
                    "status_code": upload_result.status_code,
                    "message": "Document uploaded successfully"
                }
            else:
                logging.error(f"Failed to upload image '{image_name}': {upload_result.error_message}")
                return {
                    "success": False,
                    "document_key": upload_result.key,
                    "status_code": upload_result.status_code,
                    "error": upload_result.error_message
                }
        else:
            logging.error(f"No result returned for image upload: {image_name}")
            return {
                "success": False,
                "error": "No result returned from search service"
            }
            
    except Exception as e:
        logging.error(f"Error uploading image '{image_name}' to search index: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


def search_similar_images(query_vector: List[float], top_k: int = 5) -> List[dict]:
    """
    Search for similar images using vector similarity.
    
    Args:
        query_vector: Vector embedding to search for
        top_k: Number of similar images to return
        
    Returns:
        List[dict]: List of similar images with scores
    """
    try:
        search_client = get_search_client()
        
        # Create vectorized query
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="vector"
        )
        
        # Perform search
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["id", "image_name", "image_url"],
            top=top_k
        )
        
        # Format results
        similar_images = []
        for result in results:
            similar_images.append({
                "id": result.get("id"),
                "image_name": result.get("image_name"),
                "image_url": result.get("image_url"),
                "score": result.get("@search.score", 0.0)
            })
        
        logging.info(f"Found {len(similar_images)} similar images")
        return similar_images
        
    except Exception as e:
        logging.error(f"Error searching for similar images: {str(e)}")
        return []


def delete_image(image_name: str) -> dict:
    """
    Delete an image document from the search index.
    
    Args:
        image_name: Name/identifier of the image to delete
        
    Returns:
        dict: Deletion result
    """
    try:
        search_client = get_search_client()
        
        # Create document ID (same logic as upload)
        document_id = generate_document_id(image_name)
        
        # Delete document
        result = search_client.delete_documents([{"id": document_id}])
        
        if result and len(result) > 0:
            delete_result = result[0]
            if delete_result.succeeded:
                logging.info(f"Successfully deleted image '{image_name}' from search index")
                return {
                    "success": True,
                    "document_key": delete_result.key,
                    "status_code": delete_result.status_code,
                    "message": "Document deleted successfully"
                }
            else:
                logging.error(f"Failed to delete image '{image_name}': {delete_result.error_message}")
                return {
                    "success": False,
                    "document_key": delete_result.key,
                    "status_code": delete_result.status_code,
                    "error": delete_result.error_message
                }
        else:
            return {
                "success": False,
                "error": "No result returned from search service"
            }
            
    except Exception as e:
        logging.error(f"Error deleting image '{image_name}' from search index: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }
