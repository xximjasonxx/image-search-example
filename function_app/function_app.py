import logging
import json
import azure.functions as func
from functions import (
    extract_blob_url_from_subject,
    process_image_complete,
    is_image_file
)
from search import upload, search_similar_images
from llm import vectorize_text

app = func.FunctionApp()

@app.function_name(name="process_blob")
@app.event_grid_trigger(arg_name="event")
def process_blob(event: func.EventGridEvent):
    """Process blob creation events and vectorize images using Azure AI Vision."""
    try:
        # Log the incoming event
        event_data = {
            'id': event.id,
            'data': event.get_json(),
            'topic': event.topic,
            'subject': event.subject,
            'event_type': event.event_type,
        }
        logging.info('Processing EventGrid event: %s', json.dumps(event_data, indent=2))
        
        # Check if this is a blob creation event for an image
        if event.event_type != "Microsoft.Storage.BlobCreated":
            logging.info(f"Skipping non-blob-creation event: {event.event_type}")
            return
        
        # Extract blob URL from event subject
        try:
            blob_url = extract_blob_url_from_subject(event.subject)
            logging.info(f"Extracted blob URL: {blob_url}")
        except ValueError as e:
            logging.error(f"Failed to extract blob URL: {str(e)}")
            return
        
        # Check if the blob is an image (basic check by file extension)
        if not is_image_file(blob_url):
            logging.info(f"Skipping non-image file: {blob_url}")
            return
        
        # Initialize Azure AI Vision client
        try:
            # Process the image (both vectorization and analysis)
            processing_result = process_image_complete(blob_url)
            logging.info(f"Successfully processed image: {blob_url}")
            logging.info(f"Vector embedding length: {len(processing_result.get('vector_embedding', []))}")
            logging.info(f"Analysis result: {json.dumps(processing_result.get('analysis', {}), indent=2)}")
            
            # Extract image name from the blob URL
            image_name = blob_url.split('/')[-1]  # Get filename from URL
            
            # Upload to search index
            try:
                upload_result = upload(
                    image_name=image_name,
                    url=blob_url,
                    vector=processing_result.get('vector_embedding', [])
                )
                
                if upload_result.get('success'):
                    logging.info(f"Successfully uploaded image '{image_name}' to search index")
                    logging.info(f"Document key: {upload_result.get('document_key')}")
                else:
                    logging.error(f"Failed to upload image '{image_name}' to search index: {upload_result.get('error')}")
                
            except Exception as e:
                logging.error(f"Error uploading image '{image_name}' to search index: {str(e)}")
            
        except Exception as e:
            logging.error(f"Failed to process image {blob_url}: {str(e)}")
            return
            
    except Exception as e:
        logging.error(f"Unexpected error processing event: {str(e)}")
        raise


@app.function_name(name="search")
@app.route(route="search", methods=["POST"])
def query_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP endpoint that accepts POST requests with a JSON payload containing a 'query' field."""
    try:
        # Parse JSON body
        try:
            req_body = req.get_json()
        except ValueError:
            return func.HttpResponse(
                "Invalid JSON in request body",
                status_code=400
            )
        
        # Check if request body is None
        if req_body is None:
            return func.HttpResponse(
                "Request body is required",
                status_code=400
            )
        
        # Check if 'query' field is present
        if 'query' not in req_body:
            return func.HttpResponse(
                "Missing required field 'query' in request body",
                status_code=400
            )
        
        query = req_body['query']
        logging.info(f"Received query: {query}")
        
        # Vectorize the query text
        vector = vectorize_text(query)
        
        # Search for similar images using the vector
        similar_images = search_similar_images(vector)
        
        # Return success response
        return func.HttpResponse(
            json.dumps({
                "message": "Query received successfully", 
                "query": query,
                "similar_images": similar_images
            }),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"Error processing query request: {str(e)}")
        return func.HttpResponse(
            "Internal server error",
            status_code=500
        )