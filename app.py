from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from src.core.llm.manager import LLMManager
from src.core.state_management.state_manager import StateManager
from src.core.workflow.etl_utils import ETLUtils
from src.core.workflow.etl_agent_flow import ETLAgentFlow
from src.core.workflow.etl_state import ETLWorkflowState, ETLRequirements
import logging
import uuid
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
Path("workflow_states").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize core components
llm_manager = LLMManager()
state_manager = StateManager()
etl_utils = ETLUtils(llm_manager)
etl_agent = ETLAgentFlow(llm_manager, state_manager)

# Store active workflows
active_workflows = {}

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze(request: Request):
    """Async endpoint for analyzing ETL requirements."""
    try:
        data = await request.json()
        user_input = data.get("message", "")
        
        if not user_input:
            return JSONResponse(
                status_code=400,
                content={"error": "Message is required"}
            )
        
        # Generate a unique workflow ID
        workflow_id = str(uuid.uuid4())
        
        # Run the ETL workflow with user input
        result = await etl_agent.run(workflow_id, user_input)
        
        return JSONResponse(content={
            "workflow_id": workflow_id,
            "result": result.dict()
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 