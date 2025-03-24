# Description: This file is used to test the functionality of the agents and the supervisor agent.
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, exceptions
from datetime import datetime, timezone
import uuid
import smtplib
from email.mime.text import MIMEText
from transformers import pipeline
from typing import Tuple
from typing import List, Optional
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

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

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

def invoke_azure_openai(messages, system_prompt, max_retries=3):
    """Invoke Azure OpenAI API with error handling and rate limiting."""
    retry_count = 0
    base_delay = 1  # Initial delay in seconds

    while retry_count < max_retries:
        response = model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

# Core Functions with explicit docstrings
def get_user_data(user_id: str) -> dict:
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        if not user_docs:
            # Create a new user document with default values
            user_doc = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "question_index": 0,
                "answers": [],
                "last_assessment": None,
                "messages": [],
                "context": {},
                "next_agent": None,
                "recent_assessment_notified": False
            }
            container.create_item(user_doc)
            return user_doc
        return user_docs[0]
    except Exception as e:
        raise
    
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(summary: str) -> Tuple[str, float]:
    """Perform sentiment analysis using a pre-trained model.
    
    Args:
        summary (str): The summary of the user's input.
    
    Returns:
        Tuple[str, float]: Empathetic response and sentiment severity score (0.0 to 1.0).
    """
    # Get sentiment prediction
    result = sentiment_pipeline(summary)[0]
    label = result['label']
    score = result['score']  # Confidence score from the model

    # Customize response based on sentiment and confidence
    if label == 'NEGATIVE':
        if score > 0.9:
            return (
                "I'm really sorry you're feeling this way. You're not alone, and support is available.",
                0.8  # High severity
            )
        elif score > 0.7:
            return (
                "I understand that this might be a difficult time. Let's find ways to ease it together.",
                0.5  # Moderate severity
            )
        else:
            return (
                "It seems like you're not feeling your best. Let's explore this together—what's on your mind?",
                0.3  # Low severity
            )
    elif label == 'POSITIVE':
        return (
            "It's great to hear you're feeling this way! How can I support you further?",
            0.1  # Low severity
        )
    else:  # Neutral or other cases
        return (
            "Thanks for sharing how you feel. Let's explore this together—what's on your mind?",
            0.3  # Neutral severity
        )
    
NUMERIC_PATTERN = re.compile(r'\b([0-9]|10)\b')

def get_summary(mood: str) -> str:
    """Generate a summary of the user's mood. (Placeholder implementation)"""
    # In a real system, this could summarize longer text. Here, we just return the input.
    return mood.strip()

def wellness_check(patient_mood: str, additional_responses: Optional[List[str]] = None) -> str:
    """Assess the user's mental health with empathetic, open-ended engagement.
    
    Args:
        patient_mood (str): The user's description of their mood.
        additional_responses (Optional[List[str]]): Responses to follow-up questions. Defaults to None.
    
    Returns:
        str: A tailored, empathetic response with questions or support escalation.
    """
    # Get sentiment analysis results
    summary = get_summary(patient_mood)
    sentiment_response, severity_score = analyze_sentiment(summary)

    # If no follow-up responses are provided, ask open-ended questions
    if additional_responses is None or len(additional_responses) < 3:
        questions = [
            "What’s been weighing on your mind lately?",
            "Can you tell me more about what’s making you feel this way?",
            "How has this been affecting your day-to-day life?"
        ]
        return (
            f"{sentiment_response} To help me support you better, could you share a bit more? "
            f"Here are a few questions that might help us get to the heart of it: {' '.join(questions)}"
        )

    # Process follow-up responses
    responses_lower = [r.lower() for r in additional_responses]
    
    # Check for intensity (e.g., "I feel 7 out of 10")
    intensity = None
    for resp in responses_lower:
        match = NUMERIC_PATTERN.search(resp)
        if match:
            intensity = int(match.group())
            break

    # Escalate or tailor response based on sentiment and follow-up input
    if "yes" in responses_lower or (intensity is not None and intensity >= 7) or severity_score >= 0.8:
        return (
            "It sounds like you’re going through a really hard time right now. I’m here for you, "
            "and I think it’s important to get some extra support. A therapist will reach out to you "
            "soon—please hang in there, you’re not alone."
        )
    elif intensity is not None and intensity >= 4 or severity_score >= 0.5:
        return (
            "I can see that you’re managing a lot right now. It’s okay to feel this way, and I’m here "
            "to help. Let’s keep working on this together—small steps can make a big difference."
        )
    else:
        return (
            "It’s great to hear that you’re doing okay. Remember, I’m always here if things change "
            "or if you just want to chat. Keep taking care of yourself!"
        )

def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
    """
    You are the Therapy Planning specialist of SoulSync AI.
    Focus on:
    1. Creating personalized therapy plans tailored to the patient's unique condition and history.
    2. Providing actionable recommendations that address specific triggers and needs.
    3. Setting realistic goals for short-term relief and long-term growth.
    4. Maintaining supportive communication to foster trust and progress.
    """
    condition_lower = condition.lower()
    chat_summary = "No chat history provided."
    past_emotions = []
    triggers = []
    severity = "moderate"  # Default severity

    # Analyze chat history for context
    if chat_history:
        user_msgs = [msg["content"].lower() for msg in chat_history if msg["role"] == "user"]
        chat_summary = "Conversation Summary:\n" + "\n".join([f"Patient: {msg}" for msg in user_msgs[-4:]])
        past_emotions = [msg for msg in user_msgs if any(word in msg for word in ["sad", "anxious", "stressed", "hopeless", "overwhelmed", "angry"])]
        triggers = [msg for msg in user_msgs if any(word in msg for word in ["work", "family", "relationship", "school", "health", "money"])]

    # Assess severity and refine condition
    if any(word in condition_lower for word in ["overwhelming", "constant", "severe", "can’t cope"]):
        severity = "high"
    elif any(word in condition_lower for word in ["a little", "sometimes", "mild"]):
        severity = "low"

    # Initialize therapy plan components
    focus_areas = []
    actionable_recommendations = []
    realistic_goals = []
    supportive_communication = [
        "Schedule weekly check-ins to monitor progress and adjust the plan as needed.",
        "Encourage open sharing of feelings and challenges in every session."
    ]

    # Dynamic therapy type and plan based on condition and history
    if "anxiety" in condition_lower or "stressed" in " ".join(past_emotions):
        therapy_type = "Cognitive Behavioral Therapy (CBT)"
        focus_areas.extend([
            "Anxiety management: Reduce the intensity and frequency of anxious episodes.",
            "Trigger identification: Understand what sparks anxiety."
        ])
        actionable_recommendations.extend([
            "Breathing exercises: Practice 10-15 minutes daily, increasing duration if severity is high.",
            "Anxiety log: Record anxious moments and their triggers daily."
        ])
        realistic_goals.extend([
            "Short-term: Reduce one anxiety episode this week by using breathing techniques.",
            "Long-term: Develop a personalized anxiety management toolkit within 6 weeks."
        ])
        if "work" in condition_lower or "work" in " ".join(triggers):
            focus_areas.append("Work-related stress: Address workplace pressures.")
            actionable_recommendations.append("Task prioritization: Spend 15 minutes daily organizing work tasks.")
        if severity == "high":
            actionable_recommendations.append("Grounding: Use the 5-4-3-2-1 method during peak anxiety.")

    elif "depression" in condition_lower or "sad" in " ".join(past_emotions) or "hopeless" in " ".join(past_emotions):
        therapy_type = "Mindfulness-Based Cognitive Therapy (MBCT)"
        focus_areas.extend([
            "Mood stabilization: Lift and sustain emotional well-being.",
            "Self-awareness: Recognize and interrupt negative thought cycles."
        ])
        actionable_recommendations.extend([
            "Mindfulness meditation: Practice 5-10 minutes daily to center yourself.",
            "Gratitude journaling: Write 3 things you’re thankful for each day."
        ])
        realistic_goals.extend([
            "Short-term: Complete one mindfulness session daily for a week.",
            "Long-term: Build a support network and establish a positive routine within a month."
        ])
        if "alone" in condition_lower or "lonely" in " ".join(triggers):
            focus_areas.append("Social connection: Strengthen interpersonal relationships.")
            actionable_recommendations.append("Outreach: Contact one friend or family member this week.")
        if severity == "high":
            supportive_communication.append("Provide immediate support options for low moments.")

    elif "anger" in condition_lower or "angry" in " ".join(past_emotions):
        therapy_type = "Anger Management Therapy"
        focus_areas.extend([
            "Emotional regulation: Control anger outbursts.",
            "Conflict resolution: Develop healthier response patterns."
        ])
        actionable_recommendations.extend([
            "Cool-down technique: Step away and count to 10 during anger triggers.",
            "Reflection: Write about anger incidents to identify patterns."
        ])
        realistic_goals.extend([
            "Short-term: Use the cool-down technique in one situation this week.",
            "Long-term: Reduce anger frequency by 50% within two months."
        ])
        if "relationship" in condition_lower or "family" in " ".join(triggers):
            focus_areas.append("Interpersonal dynamics: Improve relationship interactions.")
            actionable_recommendations.append("Communication practice: Role-play assertive responses.")

    else:
        therapy_type = "Exploratory Therapy"
        focus_areas.extend([
            "Emotional exploration: Identify underlying feelings and needs.",
            "Behavioral insight: Understand reaction patterns."
        ])
        actionable_recommendations.extend([
            "Daily reflection: Spend 5-10 minutes writing about your day and emotions.",
            "Mood tracking: Note your emotional state twice daily."
        ])
        realistic_goals.extend([
            "Short-term: Identify one recurring emotion or trigger this week.",
            "Long-term: Create a personal emotional awareness plan within a month."
        ])
        supportive_communication.append("Explore patient’s needs further in the next session.")
        if not condition_lower.strip() or len(condition_lower.split()) < 3:
            user_response = (
                "I want to create a therapy plan tailored just for you. Could you share more about what’s been on your mind lately "
                "or how you’ve been feeling? This will help me personalize it better!"
            )
            therapy_plan_text = f"Pending more patient input for personalization.\nCondition: {condition}\n{chat_summary}"
            return {
                "user_response": user_response,
                "therapy_plan": therapy_plan_text,
                "agent_name": "therapy_expert"
            }

    # Add trigger-specific personalization
    if triggers:
        trigger_summary = " ".join(triggers)
        if "family" in trigger_summary or "relationship" in trigger_summary:
            focus_areas.append("Relationship health: Address interpersonal stressors.")
            actionable_recommendations.append("Boundary setting: Practice saying 'no' in low-stakes situations.")
        if "health" in trigger_summary:
            focus_areas.append("Physical well-being: Explore health-related emotional impacts.")
            actionable_recommendations.append("Self-care: Dedicate 10 minutes daily to a health-focused activity.")

    # Construct the therapy plan text
    therapy_plan_text = (
        f"Subject: Therapy Recommendation for Patient {user_id or 'Unknown'}\n\n"
        f"Therapy Recommendation:\n"
        f"Based on our conversation, here is a concise therapy recommendation:\n\n"
        f"**Personalized Therapy Plan:**\n\n"
        f"1. **Focus Areas:**\n   - " + "\n   - ".join(focus_areas) + "\n\n"
        f"2. **Actionable Recommendations:**\n   - " + "\n   - ".join(actionable_recommendations) + "\n\n"
        f"3. **Realistic Goals:**\n   - " + "\n   - ".join(realistic_goals) + "\n\n"
        f"4. **Supportive Communication:**\n   - " + "\n   - ".join(supportive_communication) + "\n\n"
        f"Feel free to reach out if you need further clarification or adjustments to this plan!\n\n"
        f"{chat_summary}"
    )

    # User-facing response
    user_response = (
        f"I’ve crafted a personalized {therapy_type} plan for you. We’ll focus on {focus_areas[0].split(':')[0].lower()} "
        f"and {focus_areas[1].split(':')[0].lower()}. Start with {actionable_recommendations[0].split(':')[0].lower()} "
        f"this week. Your therapist has this plan and will reach out soon to support you!"
    )

    return {
        "user_response": user_response,
        "therapy_plan": therapy_plan_text,
        "agent_name": "therapy_expert"
    }

def post_rehab_followup(patient_status: str) -> str:
    """Assess and categorize the patient's post-rehab status for appropriate follow-up.
    Args:
        patient_status (str): The user's reported status after rehab.
    Returns:
        str: Guidance based on the risk level.
    """
    patient_status = patient_status.lower()
    # High Risk: Relapse, severe distress, or overwhelming struggles
    high_risk_keywords = ["relapse", "not better", "worse", "overwhelmed", "can't cope", "hopeless"]
    # Medium Risk: Struggles but manages daily life
    medium_risk_keywords = ["struggling", "some issues", "having trouble", "not fully okay", "difficult"]
    # Low Risk: Improvement noted
    low_risk_keywords = ["improved", "better", "recovering", "feeling okay", "coping"]
    if any(word in patient_status for word in high_risk_keywords):
        return "High risk detected. We strongly recommend scheduling a follow-up therapy session immediately."
    elif any(word in patient_status for word in medium_risk_keywords):
        return "Medium risk detected. You might benefit from support groups and occasional check-ins with a therapist."
    elif any(word in patient_status for word in low_risk_keywords):
        return "Low risk detected. Keep practicing self-help techniques and maintain regular check-ins. You're doing great!"
    return "Thank you for sharing. If you ever need support, we are here for you."

def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
    """Send the therapy plan to the therapist via email.
    
    Args:
        therapy_plan (str): The therapy plan text to send.
        user_id (str): The user's ID for identification.
    
    Returns:
        bool: True if email sent successfully, False otherwise.
    """
    therapist_email = os.getenv("THERAPIST_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
        return False

    subject = f"Therapy Plan for User {user_id}"
    msg = MIMEText(therapy_plan)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = therapist_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        return True
    except Exception:
        return False

# Agent Creation
try:
    wellness_check_agent = create_react_agent(
        model=model,
        tools=[wellness_check],
        name="wellness_check_expert",
        prompt="You are an expert in mental health assessments. For any user message describing their mood or emotions, use the 'wellness_check' tool to assess their mental state. Pass the user's message as the 'patient_mood' parameter and, if provided, any additional responses as 'additional_responses'. Return the tool's output as your response."
    )
except Exception as e:
    raise

try:
    therapy_agent = create_react_agent(
        model=model,
        tools=[provide_therapy, send_therapy_plan_to_therapist],
        name="therapy_expert",
        prompt="You are the Personalized Therapy Agent, part of SoulSync AI. When a user requests therapy or indicates a need for professional support, use the 'provide_therapy' tool with the user's message as the 'condition' parameter, their user_id, and chat history. Return the 'user_response' from the tool’s output as your response. The 'send_therapy_plan_to_therapist' tool will handle sending the therapy plan to the therapist and it is compulsory to always send a detailed therapy plan to the therapist when the user needs it with a brief chat summary."
    )
except Exception as e:
    raise

try:
    post_rehab_agent = create_react_agent(
        model=model,
        tools=[post_rehab_followup],
        name="post_rehab_expert",
        prompt=(
            "You specialize in helping patients after therapy. Use the 'post_rehab_followup' tool "
            "to analyze the user's post-rehab status by passing their message as 'patient_status'. "
            "Categorize their risk level (low, medium, high) and provide an appropriate response."
        )
    )
except Exception as e:
    raise

# Create Supervisor Agent
try:
    chat_agent = create_supervisor(
        [wellness_check_agent, therapy_agent, post_rehab_agent],
        model=model,
        prompt=(
            "Your role is to:\n"
            "1. Understand the user's needs and emotions based on their message and conversation history.\n"
            "2. Provide an empathetic, natural response without mentioning the routing process.\n" 
            "Route to appropriate agent based on intent: 'therapy'->therapy_agent, 'wellness'->wellness_check_agent, 'rehab'->post_rehab_agent, else->chat_agent."
            "**Instructions**:\n"
            "- Prioritize explicit keywords: 'therapy' or 'therapist' -> therapy intent; emotional distress -> wellness intent.\n"
            "- Use conversation history to understand emotional state.\n"
            "- Be strict about intent classification."
            "Do not change the response the other agents are generating , present it to the user as it is and take the conversation ahead until completed.Give user the response from wellness_check_expert as it is do not interpreat it."
        )
    )
except Exception as e:
    raise

# Compile the workflow
app_ai = chat_agent.compile()

# Cosmos DB Helper Functions
def get_user_data(user_id: str) -> dict:
    """Retrieve user data from Cosmos DB."""
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        return user_docs[0] if user_docs else {}
    except Exception as e:
        raise

def store_user_data(user_id: str, messages: list, context: dict) -> None:
    """Store or update user data in Cosmos DB."""
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            user_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "messages": [],
                "context": {},
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        user_data["messages"] = messages
        user_data["context"] = context
        user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        container.upsert_item(user_data)
    except Exception as e:
        raise

# API Routes
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
    existing_users = list(container.query_items(query, enable_cross_partition_query=True))
    if existing_users:
        return jsonify({"error": "Email already exists"}), 400

    user_id = str(uuid.uuid4())
    user_doc = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        **data,
        "messages": [],
        "context": {},
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    container.create_item(user_doc)
    return jsonify({"message": "Signup successful", "user_id": user_id}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing email or password"}), 400

    query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
    users = list(container.query_items(query, enable_cross_partition_query=True))
    
    if not users or users[0]["password"] != data["password"]:
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"messages": []}), 200
        messages = user_data.get("messages", [])
        return jsonify({"messages": messages})
    except Exception as e:
        return jsonify({"error": "Failed to fetch chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "")
    additional_responses = data.get("additional_responses", [])
   
    if not user_message or not user_id:
        return jsonify({"error": "Missing user_id or message"}), 400
   
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({"error": "User not found"}), 404

        messages = user_data.get("messages", [])
        context = user_data.get("context", {})
        
        messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        last_user_messages = [msg["content"] for msg in messages[-3:] if msg["role"] == "user"]
        if len(last_user_messages) >= 2 and all(msg == last_user_messages[-1] for msg in last_user_messages[-2:]):
            response = "I notice you've said the same thing a few times. Can you tell me more about how you’re feeling?"
            agent_used = "chat_agent"
        else:
            langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
            response_data = {
                "messages": langchain_messages,
                "additional_responses": additional_responses,
                "user_id": user_id,
                "chat_history": messages
            }
            
            max_iterations = 10
            iteration = 0
            final_response = None
            agent_used = "chat_agent"  # Default

            while iteration < max_iterations:
                response_data = app_ai.invoke(response_data)
                iteration += 1
                next_agent = response_data.get("next_agent")

                # Check the last message for tool output or agent response
                last_message = response_data["messages"][-1]

                if hasattr(last_message, "content") and last_message.content:  # AIMessage with content
                    final_response = last_message.content
                    agent_used = getattr(last_message, "name", "chat_agent")
                elif last_message.get("role") == "tool" and last_message.get("content"):  # Tool output
                    final_response = last_message["content"]
                    agent_used = response_data["messages"][-2].get("name", "Unknown Agent")  # Agent that called the tool
                elif isinstance(last_message, dict) and "user_response" in last_message:  # Therapy agent dict
                    final_response = last_message["user_response"]
                    agent_used = last_message.get("agent_name", "Unknown Agent")

                if not next_agent or iteration == max_iterations:
                    break

            if iteration >= max_iterations:
                final_response = "I seem to be having trouble processing your request. Can you tell me more?"
                agent_used = "chat_agent"

            response = final_response

            # Handle therapy agent email sending
            if agent_used == "therapy_expert" and isinstance(last_message, dict) and "therapy_plan" in last_message:
                success = send_therapy_plan_to_therapist(last_message["therapy_plan"], user_id)
                if not success:
                    response += " (Note: There was an issue sending the plan to your therapist; we’ll retry later.)"

        messages.append({
            "role": "assistant",
            "content": response,
            "agent_name": agent_used,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        store_user_data(user_id, messages, context)
        return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/welcome", methods=["POST"])
def welcome():
    return jsonify({"response": "Welcome to SoulSync! How can I assist you today?"})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)




# # Description: This file is used to test the functionality of the agents and the supervisor agent.
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from langchain_openai import AzureChatOpenAI
# from langgraph_supervisor import create_supervisor
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv
# from azure.cosmos import CosmosClient, exceptions
# from datetime import datetime, timezone
# import uuid
# import smtplib
# from email.mime.text import MIMEText
# from typing import List, Optional
# import re

# # Load environment variables
# load_dotenv()

# app = Flask(__name__)
# CORS(app)

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

# # Initialize Azure OpenAI Model
# model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15",
#     deployment_name="gpt-4o-mini"
# )

# def invoke_azure_openai(messages, system_prompt, max_retries=3):
#     """Invoke Azure OpenAI API with error handling and rate limiting."""
#     retry_count = 0
#     base_delay = 1  # Initial delay in seconds

#     while retry_count < max_retries:
#         response = model.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": system_prompt}] + messages,
#             temperature=0.7
#         )
#         return response.choices[0].message.content.strip()

# # Core Functions with explicit docstrings
# def get_user_data(user_id: str) -> dict:
#     try:
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))
#         if not user_docs:
#             # Create a new user document with default values
#             user_doc = {
#                 "id": str(uuid.uuid4()),
#                 "user_id": user_id,
#                 "question_index": 0,
#                 "answers": [],
#                 "last_assessment": None,
#                 "messages": [],
#                 "context": {},
#                 "next_agent": None,
#                 "recent_assessment_notified": False
#             }
#             container.create_item(user_doc)
#             return user_doc
#         return user_docs[0]
#     except Exception as e:
#         raise
    
# def get_summary(user_input: str) -> str:
#     """Generate a brief summary of the user's input.
    
#     Args:
#         user_input (str): The user's description of their experience.
    
#     Returns:
#         str: A summary string prefixed with 'User is experiencing:'.
#     """
#     return f"User is experiencing: {user_input}"

# # Precompiled regex for numeric extraction (improves efficiency)
# NUMERIC_PATTERN = re.compile(r'\d+')

# # Keyword sets for sentiment analysis (faster lookups with sets)
# SEVERE_KEYWORDS = {"hopeless", "overwhelmed", "despair", "suicidal", "exhausted"}
# STRESS_KEYWORDS = {"stressed", "anxious", "nervous", "tense", "worried"}
# POSITIVE_KEYWORDS = {"happy", "good", "calm", "peaceful", "better"}

# def get_summary(user_input: str) -> str:
#     """Generate a concise summary of the user's input, trimming unnecessary whitespace.
    
#     Args:
#         user_input (str): The user's description of their experience.
    
#     Returns:
#         str: A trimmed summary prefixed with 'User is experiencing:'.
#     """
#     return f"User is experiencing: {user_input.strip()}"

# def analyze_sentiment(summary: str) -> tuple[str, float]:
#     """Perform enhanced sentiment analysis with severity scoring.
    
#     Args:
#         summary (str): The summary of the user's input.
    
#     Returns:
#         tuple[str, float]: Empathetic response and sentiment severity score (0.0 to 1.0).
#     """
#     summary_lower = summary.lower()
    
#     # Check for severe distress
#     if any(keyword in summary_lower for keyword in SEVERE_KEYWORDS):
#         return (
#             "I'm really sorry you're feeling this way. You're not alone, and support is available.",
#             0.8  # High severity
#         )
#     # Check for stress/anxiety
#     elif any(keyword in summary_lower for keyword in STRESS_KEYWORDS):
#         return (
#             "I understand that stress can be a lot to handle. Let's find ways to ease it together.",
#             0.5  # Moderate severity
#         )
#     # Check for positive sentiment
#     elif any(keyword in summary_lower for keyword in POSITIVE_KEYWORDS):
#         return (
#             "It's great to hear you're feeling this way! How can I support you further?",
#             0.1  # Low severity
#         )
#     # Default for neutral or unspecified mood
#     return (
#         "Thanks for sharing how you feel. Let's explore this together—what's on your mind?",
#         0.3  # Neutral severity
#     )

# def wellness_check(patient_mood: str, additional_responses: Optional[List[str]] = None) -> str:
#     """Assess the user's mental health with empathetic, open-ended engagement.
    
#     Args:
#         patient_mood (str): The user's description of their mood.
#         additional_responses (Optional[List[str]]): Responses to follow-up questions. Defaults to None.
    
#     Returns:
#         str: A tailored, empathetic response with questions or support escalation.
#     """
#     summary = get_summary(patient_mood)
#     sentiment_response, severity_score = analyze_sentiment(summary)

#     if additional_responses is None or len(additional_responses) < 3:
#         questions = [
#             "What's been weighing on your mind lately?",
#             "Can you tell me more about what's making you feel this way?",
#             "How has this been affecting your day-to-day life?"
#         ]
#         return (
#             f"{sentiment_response} I'm really sorry you're feeling this way—it sounds tough. "
#             f"To help me support you better, could you share a bit more? Here are a few questions "
#             f"that might help us get to the heart of it: {' '.join(questions)}"
#         )

#     responses_lower = [r.lower() for r in additional_responses]
    
#     intensity = None
#     for resp in responses_lower:
#         match = NUMERIC_PATTERN.search(resp)
#         if match:
#             intensity = int(match.group())
#             break

#     if "yes" in responses_lower or (intensity is not None and intensity >= 7) or severity_score >= 0.8:
#         return (
#             "It sounds like you're going through a really hard time right now. I'm here for you, "
#             "and I think it's important to get some extra support. A therapist will reach out to you "
#             "soon—please hang in there, you're not alone."
#         )
#     elif intensity is not None and intensity >= 4 or severity_score >= 0.5:
#         return (
#             "I can see that you're managing a lot right now. It's okay to feel this way, and I'm here "
#             "to help. Let's keep working on this together—small steps can make a big difference."
#         )
#     else:
#         return (
#             "It's great to hear that you're doing okay. Remember, I'm always here if things change "
#             "or if you just want to chat. Keep taking care of yourself!"
#         )

# # Agent Creation for wellness_check_agent
# try:
#     wellness_check_agent = create_react_agent(
#         model=model,
#         tools=[wellness_check],
#         name="wellness_check_expert",
#         prompt="You are an expert in mental health assessments. For any user message describing their mood or emotions, use the 'wellness_check' tool to assess their mental state. Pass the user's message as the 'patient_mood' parameter and, if provided, any additional responses as 'additional_responses'. Return the tool's output as your response."
#     )
# except Exception as e:
#     raise

# def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
#     """
#     You are the Therapy Planning specialist of SoulSync AI.
#     Focus on:
#     1. Creating personalized therapy plans tailored to the patient's unique condition and history.
#     2. Providing actionable recommendations that address specific triggers and needs.
#     3. Setting realistic goals for short-term relief and long-term growth.
#     4. Maintaining supportive communication to foster trust and progress.
#     """
#     condition_lower = condition.lower()
#     chat_summary = "No chat history provided."
#     past_emotions = []
#     triggers = []
#     severity = "moderate"  # Default severity

#     # Analyze chat history for context
#     if chat_history:
#         user_msgs = [msg["content"].lower() for msg in chat_history if msg["role"] == "user"]
#         chat_summary = "Conversation Summary:\n" + "\n".join([f"Patient: {msg}" for msg in user_msgs[-4:]])
#         past_emotions = [msg for msg in user_msgs if any(word in msg for word in ["sad", "anxious", "stressed", "hopeless", "overwhelmed", "angry"])]
#         triggers = [msg for msg in user_msgs if any(word in msg for word in ["work", "family", "relationship", "school", "health", "money"])]

#     # Assess severity and refine condition
#     if any(word in condition_lower for word in ["overwhelming", "constant", "severe", "can't cope"]):
#         severity = "high"
#     elif any(word in condition_lower for word in ["a little", "sometimes", "mild"]):
#         severity = "low"

#     # Initialize therapy plan components
#     focus_areas = []
#     actionable_recommendations = []
#     realistic_goals = []
#     supportive_communication = [
#         "Schedule weekly check-ins to monitor progress and adjust the plan as needed.",
#         "Encourage open sharing of feelings and challenges in every session."
#     ]

#     # Dynamic therapy type and plan based on condition and history
#     if "anxiety" in condition_lower or "stressed" in " ".join(past_emotions):
#         therapy_type = "Cognitive Behavioral Therapy (CBT)"
#         focus_areas.extend([
#             "Anxiety management: Reduce the intensity and frequency of anxious episodes.",
#             "Trigger identification: Understand what sparks anxiety."
#         ])
#         actionable_recommendations.extend([
#             "Breathing exercises: Practice 10-15 minutes daily, increasing duration if severity is high.",
#             "Anxiety log: Record anxious moments and their triggers daily."
#         ])
#         realistic_goals.extend([
#             "Short-term: Reduce one anxiety episode this week by using breathing techniques.",
#             "Long-term: Develop a personalized anxiety management toolkit within 6 weeks."
#         ])
#         if "work" in condition_lower or "work" in " ".join(triggers):
#             focus_areas.append("Work-related stress: Address workplace pressures.")
#             actionable_recommendations.append("Task prioritization: Spend 15 minutes daily organizing work tasks.")
#         if severity == "high":
#             actionable_recommendations.append("Grounding: Use the 5-4-3-2-1 method during peak anxiety.")

#     elif "depression" in condition_lower or "sad" in " ".join(past_emotions) or "hopeless" in " ".join(past_emotions):
#         therapy_type = "Mindfulness-Based Cognitive Therapy (MBCT)"
#         focus_areas.extend([
#             "Mood stabilization: Lift and sustain emotional well-being.",
#             "Self-awareness: Recognize and interrupt negative thought cycles."
#         ])
#         actionable_recommendations.extend([
#             "Mindfulness meditation: Practice 5-10 minutes daily to center yourself.",
#             "Gratitude journaling: Write 3 things you're thankful for each day."
#         ])
#         realistic_goals.extend([
#             "Short-term: Complete one mindfulness session daily for a week.",
#             "Long-term: Build a support network and establish a positive routine within a month."
#         ])
#         if "alone" in condition_lower or "lonely" in " ".join(triggers):
#             focus_areas.append("Social connection: Strengthen interpersonal relationships.")
#             actionable_recommendations.append("Outreach: Contact one friend or family member this week.")
#         if severity == "high":
#             supportive_communication.append("Provide immediate support options for low moments.")

#     elif "anger" in condition_lower or "angry" in " ".join(past_emotions):
#         therapy_type = "Anger Management Therapy"
#         focus_areas.extend([
#             "Emotional regulation: Control anger outbursts.",
#             "Conflict resolution: Develop healthier response patterns."
#         ])
#         actionable_recommendations.extend([
#             "Cool-down technique: Step away and count to 10 during anger triggers.",
#             "Reflection: Write about anger incidents to identify patterns."
#         ])
#         realistic_goals.extend([
#             "Short-term: Use the cool-down technique in one situation this week.",
#             "Long-term: Reduce anger frequency by 50% within two months."
#         ])
#         if "relationship" in condition_lower or "family" in " ".join(triggers):
#             focus_areas.append("Interpersonal dynamics: Improve relationship interactions.")
#             actionable_recommendations.append("Communication practice: Role-play assertive responses.")

#     else:
#         therapy_type = "Exploratory Therapy"
#         focus_areas.extend([
#             "Emotional exploration: Identify underlying feelings and needs.",
#             "Behavioral insight: Understand reaction patterns."
#         ])
#         actionable_recommendations.extend([
#             "Daily reflection: Spend 5-10 minutes writing about your day and emotions.",
#             "Mood tracking: Note your emotional state twice daily."
#         ])
#         realistic_goals.extend([
#             "Short-term: Identify one recurring emotion or trigger this week.",
#             "Long-term: Create a personal emotional awareness plan within a month."
#         ])
#         supportive_communication.append("Explore patient's needs further in the next session.")
#         if not condition_lower.strip() or len(condition_lower.split()) < 3:
#             user_response = (
#                 "I want to create a therapy plan tailored just for you. Could you share more about what's been on your mind lately "
#                 "or how you've been feeling? This will help me personalize it better!"
#             )
#             therapy_plan_text = f"Pending more patient input for personalization.\nCondition: {condition}\n{chat_summary}"
#             return {
#                 "user_response": user_response,
#                 "therapy_plan": therapy_plan_text,
#                 "agent_name": "therapy_expert"
#             }

#     # Add trigger-specific personalization
#     if triggers:
#         trigger_summary = " ".join(triggers)
#         if "family" in trigger_summary or "relationship" in trigger_summary:
#             focus_areas.append("Relationship health: Address interpersonal stressors.")
#             actionable_recommendations.append("Boundary setting: Practice saying 'no' in low-stakes situations.")
#         if "health" in trigger_summary:
#             focus_areas.append("Physical well-being: Explore health-related emotional impacts.")
#             actionable_recommendations.append("Self-care: Dedicate 10 minutes daily to a health-focused activity.")

#     # Construct the therapy plan text
#     therapy_plan_text = (
#         f"Subject: Therapy Recommendation for Patient {user_id or 'Unknown'}\n\n"
#         f"Therapy Recommendation:\n"
#         f"Based on our conversation, here is a concise therapy recommendation:\n\n"
#         f"**Personalized Therapy Plan:**\n\n"
#         f"1. **Focus Areas:**\n   - " + "\n   - ".join(focus_areas) + "\n\n"
#         f"2. **Actionable Recommendations:**\n   - " + "\n   - ".join(actionable_recommendations) + "\n\n"
#         f"3. **Realistic Goals:**\n   - " + "\n   - ".join(realistic_goals) + "\n\n"
#         f"4. **Supportive Communication:**\n   - " + "\n   - ".join(supportive_communication) + "\n\n"
#         f"Feel free to reach out if you need further clarification or adjustments to this plan!\n\n"
#         f"{chat_summary}"
#     )

#     # User-facing response
#     user_response = (
#         f"I've crafted a personalized {therapy_type} plan for you. We'll focus on {focus_areas[0].split(':')[0].lower()} "
#         f"and {focus_areas[1].split(':')[0].lower()}. Start with {actionable_recommendations[0].split(':')[0].lower()} "
#         f"this week. Your therapist has this plan and will reach out soon to support you!"
#     )

#     return {
#         "user_response": user_response,
#         "therapy_plan": therapy_plan_text,
#         "agent_name": "therapy_expert"
#     }

# def post_rehab_followup(patient_status: str) -> str:
#     """Assess and categorize the patient's post-rehab status for appropriate follow-up.
#     Args:
#         patient_status (str): The user's reported status after rehab.
#     Returns:
#         str: Guidance based on the risk level.
#     """
#     patient_status = patient_status.lower()
#     # High Risk: Relapse, severe distress, or overwhelming struggles
#     high_risk_keywords = ["relapse", "not better", "worse", "overwhelmed", "can't cope", "hopeless"]
#     # Medium Risk: Struggles but manages daily life
#     medium_risk_keywords = ["struggling", "some issues", "having trouble", "not fully okay", "difficult"]
#     # Low Risk: Improvement noted
#     low_risk_keywords = ["improved", "better", "recovering", "feeling okay", "coping"]
#     if any(word in patient_status for word in high_risk_keywords):
#         return "High risk detected. We strongly recommend scheduling a follow-up therapy session immediately."
#     elif any(word in patient_status for word in medium_risk_keywords):
#         return "Medium risk detected. You might benefit from support groups and occasional check-ins with a therapist."
#     elif any(word in patient_status for word in low_risk_keywords):
#         return "Low risk detected. Keep practicing self-help techniques and maintain regular check-ins. You're doing great!"
#     return "Thank you for sharing. If you ever need support, we are here for you."

# def send_therapy_plan_to_therapist(therapy_plan: str, user_id: str) -> bool:
#     """Send the therapy plan to the therapist via email.
    
#     Args:
#         therapy_plan (str): The therapy plan text to send.
#         user_id (str): The user's ID for identification.
    
#     Returns:
#         bool: True if email sent successfully, False otherwise.
#     """
#     therapist_email = os.getenv("THERAPIST_EMAIL")
#     smtp_server = os.getenv("SMTP_SERVER")
#     smtp_port = int(os.getenv("SMTP_PORT"))
#     smtp_user = os.getenv("SMTP_USER")
#     smtp_password = os.getenv("SMTP_PASSWORD")

#     if not all([therapist_email, smtp_server, smtp_port, smtp_user, smtp_password]):
#         return False

#     subject = f"Therapy Plan for User {user_id}"
#     msg = MIMEText(therapy_plan)
#     msg["Subject"] = subject
#     msg["From"] = smtp_user
#     msg["To"] = therapist_email

#     try:
#         with smtplib.SMTP(smtp_server, smtp_port) as server:
#             server.starttls()
#             server.login(smtp_user, smtp_password)
#             server.send_message(msg)
#         return True
#     except Exception:
#         return False

# # Agent Creation
# try:
#     wellness_check_agent = create_react_agent(
#         model=model,
#         tools=[wellness_check],
#         name="wellness_check_expert",
#         prompt="You are an expert in mental health assessments. For any user message describing their mood or emotions, use the 'wellness_check' tool to assess their mental state. Pass the user's message as the 'patient_mood' parameter and, if provided, any additional responses as 'additional_responses'. Return the tool's output as your response."
#     )
# except Exception as e:
#     raise

# try:
#     therapy_agent = create_react_agent(
#         model=model,
#         tools=[provide_therapy, send_therapy_plan_to_therapist],
#         name="therapy_expert",
#         prompt="You are the Personalized Therapy Agent, part of SoulSync AI. When a user requests therapy or indicates a need for professional support, use the 'provide_therapy' tool with the user's message as the 'condition' parameter, their user_id, and chat history. Return the 'user_response' from the tool's output as your response. The 'send_therapy_plan_to_therapist' tool will handle sending the therapy plan to the therapist and it is compulsory to always send a detailed therapy plan to the therapist when the user needs it with a brief chat summary."
#     )
# except Exception as e:
#     raise

# try:
#     post_rehab_agent = create_react_agent(
#         model=model,
#         tools=[post_rehab_followup],
#         name="post_rehab_expert",
#         prompt=(
#             "You specialize in helping patients after therapy. Use the 'post_rehab_followup' tool "
#             "to analyze the user's post-rehab status by passing their message as 'patient_status'. "
#             "Categorize their risk level (low, medium, high) and provide an appropriate response."
#         )
#     )
# except Exception as e:
#     raise

# # Create Supervisor Agent with modified prompt
# try:
#     chat_agent = create_supervisor(
#         [wellness_check_agent, therapy_agent, post_rehab_agent],
#         model=model,
#         prompt=(
#             "Analyze the user's message and conversation history to classify intent. "
#             "Based on the intent, route to the appropriate agent without generating your own response:\n"
#             "- 'therapy' or 'therapist' keywords -> therapy_agent\n"
#             "- Emotional distress keywords (e.g., 'sad', 'stressed', 'anxious', 'lonely', 'hopeless') -> wellness_check_agent\n"
#             "- Rehab-related keywords (e.g., 'relapse', 'rehab', 'after therapy') -> post_rehab_agent\n"
#             "- No clear intent (general queries like 'hi', 'what is this') -> Respond directly with an empathetic, natural message\n"
#             "**Instructions**:\n"
#             "- Be strict about intent classification using explicit keywords.\n"
#             "- Do not generate a response or modify agent outputs when routing to an agent; pass their output directly.\n"
#             "- Only provide your own response when no agent is routed (general intent).\n"
#             "- Use conversation history to refine intent if needed.\n"
#         )
#     )
# except Exception as e:
#     raise

# # Compile the workflow
# app_ai = chat_agent.compile()

# # Cosmos DB Helper Functions
# def get_user_data(user_id: str) -> dict:
#     """Retrieve user data from Cosmos DB."""
#     try:
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))
#         return user_docs[0] if user_docs else {}
#     except Exception as e:
#         raise

# def store_user_data(user_id: str, messages: list, context: dict) -> None:
#     """Store or update user data in Cosmos DB."""
#     try:
#         user_data = get_user_data(user_id)
#         if not user_data:
#             user_data = {
#                 "id": str(uuid.uuid4()),
#                 "user_id": user_id,
#                 "messages": [],
#                 "context": {},
#                 "created_at": datetime.now(timezone.utc).isoformat()
#             }
#         user_data["messages"] = messages
#         user_data["context"] = context
#         user_data["updated_at"] = datetime.now(timezone.utc).isoformat()
#         container.upsert_item(user_data)
#     except Exception as e:
#         raise

# # API Routes
# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.get_json()
#     required_fields = ["firstName", "lastName", "dob", "email", "password"]
    
#     if not all(field in data for field in required_fields):
#         return jsonify({"error": "Missing required fields"}), 400

#     query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
#     existing_users = list(container.query_items(query, enable_cross_partition_query=True))
#     if existing_users:
#         return jsonify({"error": "Email already exists"}), 400

#     user_id = str(uuid.uuid4())
#     user_doc = {
#         "id": str(uuid.uuid4()),
#         "user_id": user_id,
#         **data,
#         "messages": [],
#         "context": {},
#         "created_at": datetime.now(timezone.utc).isoformat()
#     }
#     container.create_item(user_doc)
#     return jsonify({"message": "Signup successful", "user_id": user_id}), 201

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.get_json()
#     if not data.get("email") or not data.get("password"):
#         return jsonify({"error": "Missing email or password"}), 400

#     query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
#     users = list(container.query_items(query, enable_cross_partition_query=True))
    
#     if not users or users[0]["password"] != data["password"]:
#         return jsonify({"error": "Invalid email or password"}), 401

#     return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

# @app.route("/get_chat_history", methods=["GET"])
# def get_chat_history():
#     user_id = request.args.get("user_id")
#     if not user_id:
#         return jsonify({"error": "Missing user_id"}), 400

#     try:
#         user_data = get_user_data(user_id)
#         if not user_data:
#             return jsonify({"messages": []}), 200
#         messages = user_data.get("messages", [])
#         return jsonify({"messages": messages})
#     except Exception as e:
#         return jsonify({"error": "Failed to fetch chat history"}), 500

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_message = data.get("message", "").strip()
#     user_id = data.get("user_id", "")
#     additional_responses = data.get("additional_responses", [])
   
#     if not user_message or not user_id:
#         return jsonify({"error": "Missing user_id or message"}), 400
   
#     try:
#         user_data = get_user_data(user_id)
#         if not user_data:
#             return jsonify({"error": "User not found"}), 404

#         messages = user_data.get("messages", [])
#         context = user_data.get("context", {})
        
#         messages.append({
#             "role": "user",
#             "content": user_message,
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })

#         # Check for repetitive messages first
#         last_user_messages = [msg["content"] for msg in messages[-3:] if msg["role"] == "user"]
#         if len(last_user_messages) >= 2 and all(msg == last_user_messages[-1] for msg in last_user_messages[-2:]):
#             response = "I notice you've said the same thing a few times. Can you tell me more about how you're feeling?"
#             agent_used = "chat_agent"
#         else:
#             # Prepare input for supervisor
#             langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
#             supervisor_input = {
#                 "messages": langchain_messages,
#                 "additional_responses": additional_responses,
#                 "user_id": user_id,
#                 "chat_history": messages
#             }
            
#             # Invoke supervisor to get routing decision
#             supervisor_response = chat_agent.invoke(supervisor_input)
#             decision = supervisor_response["messages"][-1].content.strip()

#             if "ROUTE: wellness" in decision:
#                 response = wellness_check(user_message, additional_responses)
#                 agent_used = "wellness_check_expert"
#             elif "ROUTE: therapy" in decision:
#                 therapy_output = provide_therapy(user_message, user_id, messages)
#                 response = therapy_output["user_response"]
#                 agent_used = "therapy_expert"
#                 # Send therapy plan if applicable
#                 success = send_therapy_plan_to_therapist(therapy_output["therapy_plan"], user_id)
#                 if not success:
#                     response += " (Note: There was an issue sending the plan to your therapist; we'll retry later.)"
#             elif "ROUTE: rehab" in decision:
#                 response = post_rehab_followup(user_message)
#                 agent_used = "post_rehab_expert"
#             else:
#                 response = decision  # Supervisor's own response for general intent
#                 agent_used = "chat_agent"

#         messages.append({
#             "role": "assistant",
#             "content": response,
#             "agent_name": agent_used,
#             "timestamp": datetime.now(timezone.utc).isoformat()
#         })

#         store_user_data(user_id, messages, context)
#         return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route("/welcome", methods=["POST"])
# def welcome():
#     return jsonify({"response": "Welcome to SoulSync! How can I assist you today?"})

# if __name__ == "__main__":
#     app.run(debug=True, host="127.0.0.1", port=5000)