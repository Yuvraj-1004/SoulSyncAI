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

# if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
#     raise ValueError("Missing required Cosmos DB environment variables")

# cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
# database = cosmos_client.get_database_client(COSMOS_DB_NAME)
# container = database.get_container_client(COSMOS_DB_CONTAINER)

# # Define Agents and Their Behaviors
# AGENT_CONFIG = {
#     "Chat Agent": {
#         "description": "G",
#         "rules": [
#             "If the user asks about their mood or well-being → Route to 'Wellness Check Agent'.",
#             "If the user wants a therapy plan → Route to 'Personalized Therapy Agent'.",
#             "If the user needs post-rehab support → Route to 'Post-Rehab Follow-Up Agent'."
#         ]
#     },
#     "Wellness Check Agent": {
#         "description": "Checks user sentiment and mental state.",
#         "rules": [
#             "Ask 5 psychometric questions one by one if sentiment is unclear, calculate the score then if high crisis detected ask them to contact to human, else end conversation with a motivational message.",
#             "If risk detected, present additional 5 questions.",
#             "Use a pool of 50 questions and select randomly."
#         ]
#     },
#     "Personalized Therapy Agent": {
#         "description": "Creates therapy plans based on patient history and real-time data.",
#         "rules": [
#             "Analyze past conversations and health data.",
#             "Suggest a therapy schedule or activity plan.",
#             "Provide motivational support and goal tracking."
#         ]
#     },
#     "Post-Rehab Follow-Up Agent": {
#         "description": "Engages users after rehab to prevent relapse.",
#         "rules": [
#             "Track progress and check adherence to therapy.",
#             "Detect early signs of relapse and provide support.",
#             "Send personalized motivational messages."
#         ]
#     }
# }

# # Function to Detect Intent
# def detect_intent(user_message):
#     """Uses AI to classify the user's intent."""
#     intent_prompt = [
#         {"role": "system", "content": "Classify the intent of user by doing sentiment analysis on it into: 'wellness' means user is not in a good mood or is talking about their mental state, 'therapy' means user wants to get some information about therapies the center is providing, or 'general' means user has general queries from rehab center like FAQs."},
#         {"role": "user", "content": user_message}
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=intent_prompt,
#         temperature=0.3
#     )
#     return response.choices[0].message.content.strip().lower()

# # Function to Route to the Correct Agent
# def route_to_agent(intent, user_message):
#     """Routes the request to the appropriate agent based on intent."""
#     if intent == "wellness":
#         agent = "Wellness Check Agent"
#     elif intent == "therapy":
#         agent = "Personalized Therapy Agent"
#     elif intent == "rehab":
#         agent = "Post-Rehab Follow-Up Agent"
#     else:
#         agent = "Chat Agent"

#     # Print which agent is handling the request
#     print(f"[DEBUG] Routing to {agent} for intent: {intent}")

#     # Generate response based on agent rules
#     agent_rules = "\n".join(AGENT_CONFIG[agent]["rules"])
#     response_prompt = [
#         {"role": "system", "content": f"You are {agent}. Follow these rules:\n{agent_rules}"},
#         {"role": "user", "content": user_message}
#     ]

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=response_prompt,
#         temperature=0.7
#     )

#     return agent, response.choices[0].message.content.strip()

# # Chat API
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_id = data.get("user_id")
#     user_message = data.get("message", "").strip()

#     if not user_id or not user_message:
#         return jsonify({"error": "Missing user_id or message"}), 400

#     try:
#         # Detect intent and route to the correct agent
#         intent = detect_intent(user_message)
#         agent, ai_response = route_to_agent(intent, user_message)

#         # Store conversation history
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))

#         if not user_docs:
#             return jsonify({"error": "User not found"}), 404

#         user_doc = user_docs[0]
#         conversation_history = user_doc.get("conversation_history", [])
#         conversation_history.append({"role": "user", "content": user_message})
#         conversation_history.append({"role": "assistant", "content": ai_response})

#         # Save updated conversation history
#         user_doc["conversation_history"] = conversation_history
#         container.upsert_item(user_doc)

#         return jsonify({"response": ai_response, "agent_used": agent})

#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Error: {str(e)}"}), 500

# @app.route("/")
# def home():
#     return "AI Patient Engagement Agent is running!"

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=5000)








import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from azure.cosmos import CosmosClient, exceptions
from pathlib import Path
from dotenv import load_dotenv
import uuid
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

if not all([COSMOS_DB_ENDPOINT, COSMOS_DB_KEY, COSMOS_DB_NAME, COSMOS_DB_CONTAINER]):
    raise ValueError("Missing required Cosmos DB environment variables")

cosmos_client = CosmosClient(COSMOS_DB_ENDPOINT, COSMOS_DB_KEY)
database = cosmos_client.get_database_client(COSMOS_DB_NAME)
container = database.get_container_client(COSMOS_DB_CONTAINER)

# Define Agents and Their Behaviors
AGENT_CONFIG = {
    "Chat Agent": {
        "description": "Handles FAQs related to rehab and acts as a manager agent routing to other agents when needed",
        "rules": [
            "If the user asks about their mood or well-being → Route to 'Wellness Check Agent'.",
            "If the user wants a therapy plan → Route to 'Personalized Therapy Agent'.",
            "If the user needs post-rehab support → Route to 'Post-Rehab Follow-Up Agent'."
        ]
    },
    "Wellness Check Agent": {
        "description": "Checks user sentiment and mental state.",
        "rules": [
            "Captures the input from user in numeric format also, because you can ask questions from user which require answer in numeric format",
            "Firstly it chats with user for a few minutes by asking a bit more about their day and detect sentiment of user",
            "Ask 5 psychometric questions one by one if sentiment is unclear, If risk detected present additional 5 questions, else calculate the score then if high crisis detected ask them to contact to human, else end conversation with a motivational message.",
            "Use a pool of 50 questions and select randomly."
        ]
    },
    "Personalized Therapy Agent": {
        "description": "Creates therapy plans based on patient history and real-time data.",
        "rules": [
            "Analyze past conversations and health data.",
            "Suggest a therapy schedule or activity plan.",
            "Provide motivational support and goal tracking."
        ]
    },
    "Post-Rehab Follow-Up Agent": {
        "description": "Engages users after rehab to prevent relapse.",
        "rules": [
            "Track progress and check adherence to therapy.",
            "Detect early signs of relapse and provide support.",
            "Send personalized motivational messages."
        ]
    }
}

# Sample Pool of Psychometric Questions
PSYCHOMETRIC_QUESTIONS = [
    "On a scale of 1-10, how would you rate your mood today?",
    "How often do you feel anxious or stressed in a week?",
    "Do you find it difficult to stay motivated throughout the day?",
    "How well are you sleeping at night? (1-10)",
    "Are you feeling socially connected or isolated lately?"
]

def detect_intent(user_message):
    """Uses AI to classify the user's intent."""
    intent_prompt = [
        {"role": "system", "content": "Classify the intent of user by doing sentiment analysis on it into: 'wellness' means user is not in a good mood or is talking about their mental state, 'therapy' means user wants to get some information about therapies the center is providing, or 'general' means user has general queries from rehab center like FAQs. If the user has sent numeric value in response then it can be a response to question asked by wellness check agent, then classify it as wellness agent query and ask wellness agent to act accordingly."},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=intent_prompt,
        temperature=0.3
    )
    return response.choices[0].message.content.strip().lower()

def route_to_agent(intent, user_message, user_id):
    """Routes the request to the appropriate agent based on intent."""
    if intent == "wellness":
        return handle_wellness_check(user_id, user_message)
    elif intent == "therapy":
        agent = "Personalized Therapy Agent"
    elif intent == "rehab":
        agent = "Post-Rehab Follow-Up Agent"
    else:
        agent = "Chat Agent"

    print(f"[DEBUG] Routing to {agent} for intent: {intent}")

    agent_rules = "\n".join(AGENT_CONFIG[agent]["rules"])
    response_prompt = [
        {"role": "system", "content": f"You are {agent}. Follow these rules:\n{agent_rules}"},
        {"role": "user", "content": user_message}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=response_prompt,
        temperature=0.7
    )

    return agent, response.choices[0].message.content.strip()

def handle_wellness_check(user_id, user_message):
    """Handles the wellness check process, ensuring smooth question flow."""
    query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
    user_docs = list(container.query_items(query, enable_cross_partition_query=True))

    if not user_docs:
        return "Wellness Check Agent", "User not found in database."

    user_doc = user_docs[0]
    current_question_index = user_doc.get("question_index", 0)
    answers = user_doc.get("answers", [])

    # If user is answering a question, store the response
    if current_question_index > 0:
        answers.append(user_message)

    # If 5 questions are answered, analyze responses and return summary
    if len(answers) >= 5:
        risk_level = analyze_psychometric_answers(answers)
        response = (
            "Your responses suggest a high risk. Please contact a professional immediately."
            if risk_level == "high"
            else "Thank you for your responses. Stay positive and keep taking care of yourself!"
        )

        # Reset progress
        user_doc["question_index"] = 0
        user_doc["answers"] = []
        container.upsert_item(user_doc)

        return "Wellness Check Agent", response

    # Select next question
    next_question = PSYCHOMETRIC_QUESTIONS[current_question_index]
    user_doc["question_index"] = current_question_index + 1
    user_doc["answers"] = answers
    container.upsert_item(user_doc)

    return "Wellness Check Agent", next_question

def analyze_psychometric_answers(answers):
    """Analyzes psychometric responses to determine risk level."""
    numeric_answers = []
    
    for answer in answers:
        try:
            numeric_answers.append(int(answer))
        except ValueError:
            numeric_answers.append(5)  # Default neutral value if input isn't a number

    avg_score = sum(numeric_answers) / len(numeric_answers)

    if avg_score < 4:
        return "high"
    elif avg_score < 7:
        return "moderate"
    return "low"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id or not user_message:
        return jsonify({"error": "Missing user_id or message"}), 400

    try:
        intent = detect_intent(user_message)
        agent, ai_response = route_to_agent(intent, user_message, user_id)

        return jsonify({"response": ai_response, "agent_used": agent})

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
