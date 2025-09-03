import logging
import json
import azure.functions as func
from functions import (
    get_image_analysis_client,
    extract_blob_url_from_subject,
    vectorize_image,
    is_image_file
)

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
            vision_client = get_image_analysis_client()
        except ValueError as e:
            logging.error(f"Failed to initialize Vision client: {str(e)}")
            return
        
        # Vectorize the image
        try:
            analysis_result = vectorize_image(vision_client, blob_url)
            logging.info(f"Successfully analyzed image: {blob_url}")
            logging.info(f"Analysis result: {json.dumps(analysis_result, indent=2)}")
            
            # Here you could store the analysis results in a database, search index, etc.
            # For now, we're just logging the results
            
        except Exception as e:
            logging.error(f"Failed to analyze image {blob_url}: {str(e)}")
            return
            
    except Exception as e:
        logging.error(f"Unexpected error processing event: {str(e)}")
        raise