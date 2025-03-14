import requests
import json
import traceback
from typing import List, Optional
from pydantic import BaseModel

# Define the QueryRequest class to match the server's expected format
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

def test_search_request():
    """Test the retrieve endpoint."""
    # Endpoint for the retrieval service
    endpoint = "http://rulin@a100-st-p4de24xlarge-290:45629/retrieve"
    endpoint = "http://rulin@a100-st-p4de24xlarge-23:41383/retrieve"
    
    # Create request data directly as a dictionary
    # This is the simplest and most reliable approach
    request_data = {
        "queries": ["What are transformer models?", "How does attention mechanism work?"],
        "topk": 5,
        "return_scores": True
    }
    
    # Headers for the request
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Make the POST request to the endpoint
        # Using json parameter in requests.post automatically converts dict to JSON
        # and sets the correct Content-Type header
        response = requests.post(endpoint, json=request_data, headers=headers)
        
        # Check if the request was successful
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            # Parse the JSON response
            try:
                results = response.json()
                import pdb; pdb.set_trace()
                print(f"Parsed JSON response: {json.dumps(results, indent=2)}")
            except json.JSONDecodeError:
                print("Response was not valid JSON")
                
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        print(f"Exception traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_search_request()