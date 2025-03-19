import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from langchain_openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from pathlib import Path
from dotenv import load_dotenv
import random

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

# Define Specialized Agents
def detect_sentiment(message: str) -> str:
    """Detects sentiment from user message."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": message}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip().lower()

def personalized_therapy(user_id: str, message: str) -> str:
    """Handles personalized therapy suggestions."""
    return "Based on your history, I recommend mindfulness exercises and a therapy session schedule."

def post_rehab_followup(user_id: str, message: str) -> str:
    """Handles post-rehab follow-ups."""
    return "Let's track your progress! How have you been feeling recently?"

# Creating agents
chat_agent = create_react_agent(
    model=client,
    tools=[],
    name="chat_agent",
    prompt="You are a chat agent handling FAQs and routing specialized queries."
)

wellness_agent = create_react_agent(
    model=client,
    tools=[detect_sentiment],
    name="wellness_agent",
    prompt="You assess user sentiment and ask psychometric questions if needed."
)

therapy_agent = create_react_agent(
    model=client,
    tools=[personalized_therapy],
    name="therapy_agent",
    prompt="You provide personalized therapy plans based on user history."
)

rehab_agent = create_react_agent(
    model=client,
    tools=[post_rehab_followup],
    name="rehab_agent",
    prompt="You engage users post-rehab to prevent relapse."
)

# Supervisor to route queries
workflow = create_supervisor(
    [chat_agent, wellness_agent, therapy_agent, rehab_agent],
    model=client,
    prompt="""
    You manage agents for patient engagement. Route queries appropriately:
    - Wellness-related → wellness_agent
    - Therapy-related → therapy_agent
    - Post-rehab concerns → rehab_agent
    - General queries → chat_agent
    """
)

app_graph = workflow.compile()

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()
    
    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400
    
    result = app_graph.invoke({"messages": [{"role": "user", "content": user_message}]})
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)