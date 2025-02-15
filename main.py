from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import re
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(_name_)

app = FastAPI(title="Social Analysis API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Store analysis results with timestamp
class AnalysisSession:
    def _init_(self):
        self.results = []
        self.last_update = None
        self.next_update = None

analysis_session = AnalysisSession()

# API Constants
BASE_API_URL = "https://api.langflow.astra.datastax.com"
LANGFLOW_ID = "c6a0245a-7d3d-4e72-a7df-41ab6c8d9ead"
FLOW_ID = "a4772b53-1434-4b84-b9ff-e94a18b90f9a"
APPLICATION_TOKEN = "AstraCS:LGnSLiHFKKmfvvQjoYrgtKGK:e24ed68fde731eb8072388c62e3bdf0d0d02ca7a460c69068843c2c3766e6e42"
ENDPOINT = "json-genration"

# Default message
DEFAULT_MESSAGE = "hello"

# Request Models
class Message(BaseModel):
    message: str = DEFAULT_MESSAGE

def run_flow(message: str) -> dict:
    api_url = f"{BASE_API_URL}/lf/{LANGFLOW_ID}/api/v1/run/{ENDPOINT}"
    payload = {
        "input_value": message,
        "output_type": "chat",
        "input_type": "chat",
    }
    headers = {
        "Authorization": f"Bearer {APPLICATION_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        logger.error(f"Response content: {response.text}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_and_update_data():
    """
    Scheduled task to fetch and update data
    """
    try:
        logger.info(f"Fetching new data at {datetime.now()}")
        result = run_flow(DEFAULT_MESSAGE)
        analysis_session.results = [result]
        analysis_session.last_update = datetime.now()
        analysis_session.next_update = analysis_session.last_update + timedelta(minutes=15)
        logger.info(f"Data updated successfully. Next update at {analysis_session.next_update}")
    except Exception as e:
        logger.error(f"Error in scheduled data fetch: {e}")

@app.on_event("startup")
async def startup_event():
    """
    Start the scheduler when the application starts
    """
    try:
        # Schedule the task to run every 15 minutes
        scheduler.add_job(
            fetch_and_update_data,
            trigger=IntervalTrigger(minutes=15),
            id='fetch_data_job',
            replace_existing=True
        )
        # Initial data fetch
        await fetch_and_update_data()
        # Start the scheduler
        scheduler.start()
        logger.info("Scheduler started successfully")
    except Exception as e:
        logger.error(f"Error starting scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown the scheduler when the application stops
    """
    scheduler.shutdown()
    logger.info("Scheduler shut down")

@app.post("/analyze")
async def analyze_message(message: Message):
    """
    Analyze a message using the social analysis API
    """
    try:
        result = run_flow(message.message)
        if not result or "outputs" not in result:
            raise HTTPException(status_code=500, detail="Invalid response from API")

        analysis_session.results = [result]
        analysis_session.last_update = datetime.now()
        analysis_session.next_update = analysis_session.last_update + timedelta(minutes=15)
        
        # Extract the JSON data from the message
        try:
            message_text = result.get("outputs", [])[0].get("outputs", [])[0].get("artifacts", {}).get("message", "")
            if not message_text:
                raise HTTPException(status_code=500, detail="No message found in response")

            json_match = re.search(r'json\s*\n(.*?)\n\s*', message_text, re.DOTALL)
            if not json_match:
                raise HTTPException(status_code=500, detail="Could not find JSON data in response")
                
            json_str = json_match.group(1).strip()
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["statsCards", "staticPosts", "reels", "carousels"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Missing required fields in data: {', '.join(missing_fields)}"
                )
            
            # Format the response according to the required structure
            formatted_response = {
                "session_id": FLOW_ID,
                "outputs": [{
                    "inputs": {
                        "input_value": message.message
                    },
                    "outputs": [{
                        "results": {
                            "message": {
                                "text_key": "text",
                                "data": {
                                    "timestamp": datetime.now().isoformat(),
                                    "sender": "Machine",
                                    "sender_name": "AI",
                                    "session_id": FLOW_ID,
                                    "text": message_text,
                                    "files": [],
                                    "error": False,
                                    "edit": False,
                                    "properties": {
                                        "text_color": "",
                                        "background_color": "",
                                        "edited": False,
                                        "source": {
                                            "id": "GoogleGenerativeAIModel-ll2gt",
                                            "display_name": "Google Generative AI",
                                            "source": "learnlm-1.5-pro-experimental"
                                        },
                                        "icon": "GoogleGenerativeAI",
                                        "allow_markdown": False,
                                        "positive_feedback": None,
                                        "state": "complete",
                                        "targets": []
                                    },
                                    "category": "message",
                                    "content_blocks": [],
                                    "id": f"analysis-{datetime.now().isoformat()}",
                                    "flow_id": FLOW_ID
                                }
                            }
                        },
                        "artifacts": {
                            "message": message_text,
                            "sender": "Machine",
                            "sender_name": "AI",
                            "files": [],
                            "type": "object"
                        },
                        "outputs": {
                            "message": {
                                "message": {
                                    "timestamp": datetime.now().isoformat(),
                                    "sender": "Machine",
                                    "sender_name": "AI",
                                    "session_id": FLOW_ID,
                                    "text": message_text,
                                    "files": [],
                                    "error": False,
                                    "edit": False,
                                    "properties": {
                                        "text_color": "",
                                        "background_color": "",
                                        "edited": False,
                                        "source": {
                                            "id": "GoogleGenerativeAIModel-ll2gt",
                                            "display_name": "Google Generative AI",
                                            "source": "learnlm-1.5-pro-experimental"
                                        },
                                        "icon": "GoogleGenerativeAI",
                                        "allow_markdown": False,
                                        "positive_feedback": None,
                                        "state": "complete",
                                        "targets": []
                                    },
                                    "category": "message",
                                    "content_blocks": [],
                                    "id": f"analysis-{datetime.now().isoformat()}",
                                    "flow_id": FLOW_ID
                                },
                                "type": "message"
                            }
                        },
                        "logs": {
                            "message": []
                        }
                    }]
                }]
            }
            
            return formatted_response
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Problematic JSON string: {json_str}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON data: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in analyze_message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    return {
        "message": "Social Analysis API is running",
        "last_update": analysis_session.last_update,
        "next_update": analysis_session.next_update
    }

@app.get("/analysis-data")
async def get_analysis_data():
    """
    Display the latest analysis data with session information
    """
    if not analysis_session.results:
        return {"message": "No analysis data available"}
    
    try:
        # Get the message from the nested structure
        message = analysis_session.results[-1].get("outputs", [])[0].get("outputs", [])[0].get("artifacts", {}).get("message", "")
        
        # Extract the JSON string between the json markers
        json_match = re.search(r'json\s*\n(.*?)\n\s*```', message, re.DOTALL)
        if not json_match:
            return {"message": "Could not find JSON data in the response"}
            
        json_str = json_match.group(1).strip()
        
        # Parse the JSON string
        data = json.loads(json_str)
        
        # Add session information to the response
        response_data = {
            "data": data,
            "session_info": {
                "last_update": analysis_session.last_update.isoformat() if analysis_session.last_update else None,
                "next_update": analysis_session.next_update.isoformat() if analysis_session.next_update else None,
                "time_until_next_update": str(analysis_session.next_update - datetime.now()) if analysis_session.next_update else None
            }
        }
        
        # Return the parsed JSON data with session info
        from fastapi.responses import JSONResponse
        return JSONResponse(content=response_data)
        
    except (IndexError, AttributeError, KeyError) as e:
        logger.error(f"Error accessing data structure: {e}")
        return {"message": f"Error accessing data structure: {str(e)}"}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON: {e}")
        logger.error(f"Problematic JSON string: {json_str}")
        return {"message": f"Error parsing JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"message": f"Unexpected error: {str(e)}"}

# if _name_ == "_main_":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8585)
