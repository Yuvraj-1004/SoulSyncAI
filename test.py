import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid
from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Load environment variables
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = Flask(__name__)
CORS(app)

# Azure OpenAI Configuration
client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15"
)

# Azure Cosmos DB Configuration
COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Initialize LangGraph Agents
model = AzureChatOpenAI(deployment_name="gpt-4o-mini")

def wellness_check(user_message: str) -> str:
    """Handles wellness assessment and sentiment detection."""
    return "Assessing mental wellness... [Response from Wellness Check Agent]"

def personalized_therapy(user_message: str) -> str:
    """Handles therapy recommendations."""
    return "Providing personalized therapy plan... [Response from Therapy Agent]"

def post_rehab_followup(user_message: str) -> str:
    """Handles post-rehab monitoring."""
    return "Checking recovery progress... [Response from Follow-Up Agent]"

# Define agents
wellness_agent = create_react_agent(
    model=model,
    tools=[wellness_check],
    name="wellness_agent",
    prompt="You assess mental health and detect sentiment."
)

therapy_agent = create_react_agent(
    model=model,
    tools=[personalized_therapy],
    name="therapy_agent",
    prompt="You recommend therapy plans."
)

followup_agent = create_react_agent(
    model=model,
    tools=[post_rehab_followup],
    name="followup_agent",
    prompt="You monitor post-rehab progress."
)

# Supervisor (Chat Agent) managing workflow
chat_agent = create_supervisor(
    [wellness_agent, therapy_agent, followup_agent],
    model=model,
    prompt="""
    You are the Chat Agent. Route user queries based on intent:
    - If about mental health, delegate to Wellness Agent.
    - If about therapy plans, delegate to Therapy Agent.
    - If about rehab progress, delegate to Follow-Up Agent.
    """
)

# Compile workflow
workflow = chat_agent.compile()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id")
    
    if not user_message or not user_id:
        return jsonify({"error": "Missing user_id or message"}), 400
    
    try:
        result = workflow.invoke({"messages": [{"role": "user", "content": user_message}]})
        response = result.get("response", "I'm not sure how to help with that.")
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "AI Patient Engagement Agent is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)