    # import os
    # from flask import Flask, request, jsonify
    # from flask_cors import CORS
    # from openai import AzureOpenAI
    # from azure.cosmos import CosmosClient, exceptions
    # from pathlib import Path
    # from dotenv import load_dotenv
    # import uuid

    # # Load environment variables (unchanged)
    # dotenv_path = Path(__file__).resolve().parent / ".env"
    # load_dotenv(dotenv_path=dotenv_path)

    # app = Flask(__name__)
    # CORS(app)

    # # Azure OpenAI Configuration (unchanged)
    # client = AzureOpenAI(
    #     azure_endpoint=os.getenv("ENDPOINT"),
    #     api_key=os.getenv("API_KEY"),
    #     api_version="2023-05-15"
    # )

    # # Azure Cosmos DB Configuration (unchanged)
    # COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
    # COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
    # COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
    # COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

    # if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    #     raise ValueError("Missing required Cosmos DB environment variables")

    # cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
    # database = cosmos_client.get_database_client(COSMOS_DB_NAME)
    # container = database.get_container_client(COSMOS_DB_CONTAINER)

    # # Signup endpoint (unchanged)
    # @app.route("/signup", methods=["POST"])
    # def signup():
    #     data = request.json
    #     first_name = data.get("firstName")
    #     last_name = data.get("lastName")
    #     dob = data.get("dob")
    #     email = data.get("email")
    #     password = data.get("password")  # In production, hash this!

    #     if not all([first_name, last_name, dob, email, password]):
    #         return jsonify({"error": "Missing required fields"}), 400

    #     user_id = str(uuid.uuid4())
    #     user_data = {
    #         "id": user_id,
    #         "user_id": user_id,
    #         "first_name": first_name,
    #         "last_name": last_name,
    #         "dob": dob,
    #         "email": email,
    #         "password": password,  # Store hashed password in production
    #         "conversation_history": []
    #     }

    #     try:
    #         container.upsert_item(user_data)
    #         return jsonify({"user_id": user_id, "message": "Signup successful"}), 201
    #     except exceptions.CosmosHttpResponseError as e:
    #         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

    # # Login endpoint (unchanged)
    # @app.route("/login", methods=["POST"])
    # def login():
    #     data = request.json
    #     email = data.get("email")
    #     password = data.get("password")

    #     if not email or not password:
    #         return jsonify({"error": "Missing email or password"}), 400

    #     try:
    #         query = f"SELECT * FROM c WHERE c.email = '{email}'"
    #         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

    #         if not user_docs:
    #             return jsonify({"error": "Email not found"}), 404

    #         user_doc = user_docs[0]
    #         if user_doc["password"] != password:  # In production, compare hashed passwords
    #             return jsonify({"error": "Incorrect password"}), 401

    #         return jsonify({"user_id": user_doc["user_id"], "message": "Login successful"}), 200

    #     except exceptions.CosmosHttpResponseError as e:
    #         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    #     except Exception as e:
    #         return jsonify({"error": f"Error: {str(e)}"}), 500

    # # RESTORED: Generate summary function
    # def generate_summary(conversation_history):
    #     """Generate a brief summary of the conversation."""
    #     if not conversation_history:
    #         return "No previous conversation found."
        
    #     summary_prompt = [
    #         {"role": "system", "content": "Provide a concise 3-5 line summary of the following conversation."}
    #     ]
    #     summary_prompt.extend(conversation_history[-10:])  # Limit to last 10 messages

    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=summary_prompt
    #     )
    #     return response.choices[0].message.content.strip()

    # # RESTORED: Detect intent function
    # def detect_intent(user_message, has_previous):
    #     """Use LLM to detect user intent conversationally with improved handling of negations."""
    #     intent_prompt = [
    #         {"role": "system", "content": """Classify the user's intent based on their message. Respond with one of these categories:
    #         - 'recall_past': User wants to know about or reference the previous conversation.
    #         - 'continue': User wants to continue the previous conversation.
    #         - 'start_fresh': User wants to start a new conversation.
    #         - 'general': User is making a general statement or question with no clear intent.
    #         Consider negations (e.g., 'no' followed by a request) as part of the intent. Provide only the category name."""},
    #         {"role": "user", "content": user_message}
    #     ]
        
    #     if has_previous:
    #         intent_prompt.append({"role": "system", "content": "Note: There is a previous conversation available."})
    #     else:
    #         intent_prompt.append({"role": "system", "content": "Note: There is no previous conversation available."})

    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=intent_prompt,
    #         temperature=0.3
    #     )
    #     intent = response.choices[0].message.content.strip()

    #     if "no" in user_message.lower() and intent not in ["recall_past", "start_fresh"]:
    #         fallback_prompt = intent_prompt + [
    #             {"role": "system", "content": "Re-evaluate the intent, considering 'no' as a potential indicator to reverse the previous action or recall past context. Respond with the category name."}
    #         ]
    #         fallback_response = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=fallback_prompt,
    #             temperature=0.3
    #         )
    #         intent = fallback_response.choices[0].message.content.strip()

    #     return intent

    # # RESTORED: Get chat response function
    # def get_chat_response(user_message, conversation_history, user_id, intent):
    #     """Generate a response with a system prompt based on detected intent."""
    #     if intent == "recall_past":
    #         system_prompt = [
    #             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Summarize the user's past conversation if available, then assist. The user's ID is {user_id}. Use their chat history to recall past vibes. If no relevant history, focus on the new request."}
    #         ]
    #     elif intent == "continue":
    #         system_prompt = [
    #             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Continue the user's previous conversation naturally. The user's ID is {user_id}. Use their chat history to pick up where we left off."}
    #         ]
    #     elif intent == "start_fresh":
    #         system_prompt = [
    #             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Start a new conversation. The user's ID is {user_id}. Ignore past chat history."}
    #         ]
    #     else:  # general
    #         system_prompt = [
    #             {"role": "system", "content": f"You are SoulSync, a chill AI assistant from the future. Respond naturally to the user's new request. The user's ID is {user_id}. Only use chat history if they ask about it explicitly (e.g., 'what did we talk about?', 'remember last chat'). Avoid summarizing past conversations unless asked."}
    #         ]

    #     messages = system_prompt
    #     if intent in ["recall_past", "continue"] and conversation_history:
    #         messages.extend(conversation_history[-10:])
    #     messages.append({"role": "user", "content": user_message})

    #     response = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=messages,
    #         temperature=0.7,
    #         frequency_penalty=0.5,
    #         presence_penalty=0.5
    #     )
    #     return response.choices[0].message.content.strip()

    # # CHANGED: Updated chat endpoint to use intelligent responses
    # @app.route("/chat", methods=["POST"])
    # def chat():
    #     data = request.json
    #     user_id = data.get("user_id")
    #     user_message = data.get("message", "").strip()

    #     if not user_id or not user_message:
    #         return jsonify({"error": "Missing user_id or message"}), 400

    #     try:
    #         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
    #         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

    #         if not user_docs:
    #             return jsonify({"error": "User not found"}), 404

    #         user_doc = user_docs[0]
    #         conversation_history = user_doc.get("conversation_history", [])
    #         has_previous = bool(conversation_history)

    #         # Use intent detection
    #         intent = detect_intent(user_message, has_previous)

    #         # Generate intelligent response
    #         ai_response = get_chat_response(user_message, conversation_history, user_id, intent)

    #         # Append new messages to conversation history
    #         conversation_history.append({"role": "user", "content": user_message})
    #         conversation_history.append({"role": "assistant", "content": ai_response})

    #         # Update user document with summary if recalling past
    #         user_doc["conversation_history"] = conversation_history
    #         user_doc["summary"] = generate_summary(conversation_history) if intent == "recall_past" else user_doc.get("summary", "No summary needed")
    #         container.upsert_item(user_doc)

    #         return jsonify({"response": ai_response})

    #     except exceptions.CosmosHttpResponseError as e:
    #         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    #     except Exception as e:
    #         return jsonify({"error": f"Error: {str(e)}"}), 500

    # @app.route("/")
    # def home():
    #     return "SoulSync API is running!"

    # if __name__ == "__main__":
    #     app.run(debug=True, host="127.0.0.1", port=5000)

















# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from openai import AzureOpenAI
# from azure.cosmos import CosmosClient, exceptions
# from pathlib import Path
# from dotenv import load_dotenv
# import uuid

# # Load environment variables
# dotenv_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(dotenv_path=dotenv_path)

# app = Flask(__name__)
# CORS(app)

# # Azure OpenAI Configuration
# client = AzureOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15"
# )

# # Azure Cosmos DB Configuration
# COSMOS_DB_ENDPOINT = os.getenv("COSMOS_DB_ENDPOINT")
# COSMOS_DB_KEY = os.getenv("COSMOS_DB_KEY")
# COSMOS_DB_NAME = os.getenv("COSMOS_DB_NAME")
# COSMOS_DB_CONTAINER = os.getenv("COSMOS_DB_CONTAINER")

# cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
# database = cosmos_client.get_database_client(COSMOS_DB_NAME)
# container = database.get_container_client(COSMOS_DB_CONTAINER)

# # Function to detect user intent
# def detect_intent(user_message):
#     intent_prompt = [
#         {"role": "system", "content": "Classify the user's intent based on their message. Respond with one of these categories: \n"
#          "- 'wellness_check': User wants a wellness check.\n"
#          "- 'personalized_therapy': User wants a personalized therapy session.\n"
#          "- 'post_rehab_followup': User wants a follow-up after rehab.\n"
#          "- 'general': User has a general query."},
#         {"role": "user", "content": user_message}
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=intent_prompt,
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip()

# # Function to get response from the specific agent
# def get_agent_response(user_message, agent_type):
#     agent_prompts = {
#         "wellness_check": "You are a wellness check agent. Assess the user's mental state and ask psychometric questions if needed.",
#         "personalized_therapy": "You are a therapy agent. Provide personalized therapy advice based on user concerns.",
#         "post_rehab_followup": "You are a post-rehab follow-up agent. Guide the user through recovery progress.",
#         "general": "You are a general assistant. Answer the user's general inquiries."
#     }
    
#     messages = [
#         {"role": "system", "content": agent_prompts.get(agent_type, agent_prompts["general"])},
#         {"role": "user", "content": user_message}
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         temperature=0.7
#     )
#     return response.choices[0].message.content.strip()

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_message = data.get("message", "").strip()

#     if not user_message:
#         return jsonify({"error": "Message is required"}), 400
    
#     intent = detect_intent(user_message)
#     ai_response = get_agent_response(user_message, intent)
    
#     return jsonify({"intent": intent, "response": ai_response})

# @app.route("/")
# def home():
#     return "Patient Engagement Agent API is running!"

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=5000)




import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
from enum import Enum
from typing import Dict, Any, Tuple
import uuid

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

if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    raise ValueError("Missing required Cosmos DB environment variables")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Define Agent States
class AgentState(Enum):
    CHAT = "Chat Agent"
    WELLNESS = "Wellness Check Agent"
    THERAPY = "Personalized Therapy Agent"
    REHAB = "Post-Rehab Follow-Up Agent"

# Agent Configuration with Supervisor Graph
class AgentSupervisor:
    def __init__(self):
        self.agent_config = {
            AgentState.CHAT: {
                "description": "Routes the conversation based on user's intent",
                "rules": [
                    "If mood/well-being question → WELLNESS",
                    "If therapy plan requested → THERAPY",
                    "If post-rehab support needed → REHAB"
                ],
                "transitions": {
                    "wellness": AgentState.WELLNESS,
                    "therapy": AgentState.THERAPY,
                    "rehab": AgentState.REHAB
                }
            },
            AgentState.WELLNESS: {
                "description": "Checks user sentiment and mental state",
                "rules": [
                    "Ask 5 psychometric questions one by one if sentiment unclear",
                    "If risk detected, ask 5 more questions",
                    "Random selection from 50 questions pool"
                ],
                "transitions": {
                    "default": AgentState.CHAT
                }
            },
            AgentState.THERAPY: {
                "description": "Creates therapy plans",
                "rules": [
                    "Analyze past conversations and health data",
                    "Suggest therapy schedule/activity plan",
                    "Provide motivational support"
                ],
                "transitions": {
                    "default": AgentState.CHAT
                }
            },
            AgentState.REHAB: {
                "description": "Post-rehab follow-up support",
                "rules": [
                    "Track progress and therapy adherence",
                    "Detect relapse signs and provide support",
                    "Send motivational messages"
                ],
                "transitions": {
                    "default": AgentState.CHAT
                }
            }
        }
        self.current_state = AgentState.CHAT

    def detect_intent(self, message: str) -> str:
        """Classify user's intent using AI"""
        intent_prompt = [
            {"role": "system", "content": "Classify intent as: 'wellness', 'therapy', 'rehab', or 'general'"},
            {"role": "user", "content": message}
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=intent_prompt,
            temperature=0.3
        )
        return response.choices[0].message.content.strip().lower()

    def transition(self, intent: str) -> AgentState:
        """Determine next state based on current state and intent"""
        transitions = self.agent_config[self.current_state]["transitions"]
        if intent in transitions:
            self.current_state = transitions[intent]
        elif "default" in transitions:
            self.current_state = transitions["default"]
        return self.current_state

    def process_message(self, message: str) -> Tuple[AgentState, str]:
        """Process message and return agent response"""
        intent = self.detect_intent(message)
        current_agent = self.transition(intent)
        
        agent_rules = "\n".join(self.agent_config[current_agent]["rules"])
        response_prompt = [
            {"role": "system", "content": f"You are {current_agent.value}. Rules:\n{agent_rules}"},
            {"role": "user", "content": message}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=response_prompt,
            temperature=0.7
        )
        
        print(f"[DEBUG] Agent: {current_agent.value} | Intent: {intent}")
        return current_agent, response.choices[0].message.content.strip()

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        # Initialize supervisor for this request
        supervisor = AgentSupervisor()
        agent, ai_response = supervisor.process_message(user_message)

        # Store conversation history
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))

        if not user_docs:
            return jsonify({"error": "User not found"}), 404

        user_doc = user_docs[0]
        conversation_history = user_doc.get("conversation_history", [])
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": ai_response})

        user_doc["conversation_history"] = conversation_history
        container.upsert_item(user_doc)

        return jsonify({"response": ai_response, "agent_used": agent.value})

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "AI Patient Engagement Agent is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)