# import os
# from flask import Flask, request, jsonify
# from langchain_openai import AzureChatOpenAI
# from langgraph_supervisor import create_supervisor
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize Flask app

# # Initialize Azure OpenAI Model
# model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("ENDPOINT"),
#     api_key=os.getenv("API_KEY"),
#     api_version="2023-05-15",
#     deployment_name="gpt-4o-mini"
# )

# # ========== Define Agent Functionalities ========== #


# def get_summary(user_input: str) -> str:
#     """Generate a brief summary of the user's experience."""
#     print(f"[WELLNESS_CHECK_AGENT] Summarizing user's concern: {user_input}")
#     return f"User is experiencing: {user_input}"


# def analyze_sentiment(summary: str) -> str:
#     """Perform sentiment analysis and provide an empathetic response."""
#     print("[WELLNESS_CHECK_AGENT] Analyzing sentiment...")
#     if "hopeless" in summary or "overwhelmed" in summary:
#         return "I'm really sorry you're feeling this way. You're not alone, and support is available."
#     elif "stressed" in summary or "anxious" in summary:
#         return "I understand that stress can be overwhelming. Taking small steps can really help."
#     return "It's okay to have tough days. Remember, you're stronger than you think!"


# def generate_questions(summary: str) -> list:
#     """Generate five context-based questions if the summary is insufficient."""
#     print("[WELLNESS_CHECK_AGENT] Generating dynamic questions...")
#     return [
#         f"Can you describe more about how {summary.split(':')[-1]} affects your daily life?",
#         "Do you experience mood swings frequently? (Yes/No)",
#         "Have you had trouble sleeping lately? (Yes/No)",
#         "Do you feel disconnected from loved ones? (Yes/No)",
#         "On a scale of 1-10, how would you rate your emotional well-being?"
#     ]


# def wellness_check(patient_mood: str) -> str:
#     """Assess the patient's mental health based on mood description."""
#     print(f"[WELLNESS_CHECK_AGENT] Assessing mood: {patient_mood}")
#     summary = get_summary(patient_mood)
#     sentiment_response = analyze_sentiment(summary)
#     user_agree = input(f"[WELLNESS_CHECK_AGENT] {sentiment_response} Would you be comfortable answering a few more questions? (Yes/No)\nYou: ")
    
#     if user_agree.lower() == "yes":
#         questions = generate_questions(summary)
#         responses = []
#         for question in questions:
#             user_response = input(f"[WELLNESS_CHECK_AGENT] {question}\nYou: ")
#             responses.append(user_response.lower())
        
#         if "yes" in responses or any(int(x) >= 7 for x in responses if x.isdigit()):
#             return "High risk detected. Our therapist will contact you soon."
#         elif any(int(x) >= 4 for x in responses if x.isdigit()):
#             return "Medium risk detected. Remember, tough times don't last. You are stronger than you think!"
#         else:
#             return "Low risk detected. Keep going! Every storm runs out of rain."
    
#     return "Thank you for sharing. If you ever need support, don't hesitate to reach out."

# def provide_therapy(condition: str) -> str:
#     """Recommend therapy based on the patient's condition."""
#     if "anxiety" in condition.lower():
#         return "Recommended therapy: Cognitive Behavioral Therapy (CBT)."
#     elif "depression" in condition.lower():
#         return "Recommended therapy: Mindfulness-Based Therapy."
#     return "General therapy session is advised."

# def post_rehab_followup(patient_status: str) -> str:
#     """Engage with patient after therapy completion."""
#     if "improved" in patient_status.lower():
#         return "Continue with self-help practices and regular check-ins."
#     return "Schedule a follow-up therapy session."

# # ========== Create Agents ========== #
# wellness_check_agent = create_react_agent(
#     model=model,
#     tools=[wellness_check],
#     name="wellness_check_agent",
#     prompt="You are an expert in mental health assessments. Check the patient's mood by asking them a to tell you a brief summary about their mood, then analyze it using sentiment analysis and conduct a risk assessment if needed."
# )

# therapy_agent = create_react_agent(
#     model=model,
#     tools=[provide_therapy],
#     name="personalized_therapy_agent",
#     prompt="You are the Personalized Therapy Agent, a part of SoulSync AI, a compassionate and intelligent mental health assistant. "
#         "Your role is to assist users who have explicitly requested therapy or indicated a need for professional mental health support. "
#         "Your goal is to create a personalized therapy plan tailored to the user's emotional state, needs, and conversation history, "
#         "send this plan to the user's therapist via email for further evaluation, and provide a summarized response to the user confirming that the plan has been shared. "
#         "### Your Responsibilities: "
#         "1. **Acknowledge the User's Request**: Internally acknowledge the user's request for therapy to understand their needs. "
#         "2. **Analyze the Conversation History**: Review the user's recent messages to understand their emotional state, concerns, and triggers. "
#         "3. **Assess the User's Needs**: Identify primary mental health needs based on conversation history. "
#         "4. **Create a Personalized Therapy Plan**: Develop a therapy plan with 3-5 actionable steps tailored to the user's needs, including short-term strategies, long-term goals, and coping mechanisms. "
#         "5. **Prepare the Therapy Plan for Email**: Format the therapy plan for sending to the therapist, ensuring clarity and completeness."
# )

# post_rehab_agent = create_react_agent(
#     model=model,
#     tools=[post_rehab_followup],
#     name="post_rehab_follow_up_agent",
#     prompt="You help patients post-therapy. Engage with them and suggest the next steps."
# )

# # ========== Create Chat Agent (Supervisor) ========== #
# chat_agent = create_supervisor(
#     [wellness_check_agent, therapy_agent, post_rehab_agent],
#     model=model,
#     prompt=(
#         "You are team supervisor managing wellness-check, post-rehab and therapies expert"
#         "For mental health realted issues, use wellness_check_agent"
#         "For therapy realted issues, use therapy_agent"
#         "For post rehab realted issues, use post_rehab_agent"
#         # "Your role is to route control to correct agent only"
#         # "Your role is to:\n"
#         # "1. Understand the user's needs and emotions based on their message and conversation history.\n"
#         # "2. Provide an empathetic, natural response without mentioning the routing process.\n"
#         # "3. Classify the user's intent into one of the following categories and determine the appropriate agent to route to:\n"
#         # "   - Intent: wellness (for messages indicating mental suppression, emotional distress, or a need for mood assessment.\n"
#         # "     Examples: 'I am not well', 'I feel sad', 'I’m feeling down', 'I feel depressed', 'I am feeling sad',\n"
#         # "     'I had a fight with my family', 'I am frustrated', 'I feel overwhelmed', 'I am stressed')\n"
#         # "     - Route to: wellness_check_agent\n"
#         # "   - Intent: therapy (for messages explicitly requesting therapy or indicating a need for professional mental health support.\n"
#         # "     Examples: 'I need therapy', 'I want to talk to a therapist', 'I need professional help', 'I want a therapy plan')\n"
#         # "     - Route to: personalized_therapy_agent\n"
#         # "   - Intent: rehab (for messages related to post-rehabilitation support, relapse, or recovery.\n"
#         # "     Examples: 'I relapsed', 'I need rehab support', 'I’m struggling after rehab')\n"
#         # "     - Route to: post_rehab_follow_up_agent\n"
#         # "   - Intent: general (for all other messages that don't fit the above categories.\n"
#         # "     Examples: 'How are you?', 'Tell me about SoulSync', 'hii', 'why', 'what is this')\n"
#         # "     - Route to: None (no further routing needed)\n"
#         # "**Important Instructions**:\n"
#         # "- Prioritize explicit keywords:\n"
#         # "  - If the message contains 'therapy', 'therapist', or 'therapy plan', classify as 'therapy' and route accordingly.\n"
#         # "  - If the message contains emotional distress phrases, classify as 'wellness' and route accordingly.\n"
#         # "- Use conversation history to understand emotional state.\n"
#         # "- Be strict about following the intent classification rules."
#     )
# )

# # Compile the workflow
# app = chat_agent.compile()

# # ========== Flask API for Frontend Integration ========== #
# # ========== Interactive Chat Mode ========== #
# def chat_with_agents():
#     print("\nWelcome to the AI Healthcare Assistant! Type 'exit' to end the chat.\n")
#     while True:
#         user_query = input("You: ")
#         if user_query.lower() == "exit":
#             print("Goodbye!")
#             break
        
#         print("\n" + "="*50)
#         print(f"[TEST] User Query: {user_query}")
#         print("="*50)
        
#         try:
#             response = app.invoke({"messages": [{"role": "user", "content": user_query}]})
            
#             # Extracting agent responses and formatting output
#             messages = response.get("messages", [])
#             formatted_response = "\n[SYSTEM RESPONSE]\n"
            
#             for message in messages:
#                 agent_name = getattr(message, "name", "Unknown Agent")
#                 content = getattr(message, "content", "No response.")
#                 formatted_response += f"[{agent_name}]: {content}\n"
            
#             print(formatted_response)
#         except Exception as e:
#             print("[ERROR]:", str(e))

# # Start interactive chat
# chat_with_agents()


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
import logging
import smtplib
from email.mime.text import MIMEText

# Setup logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# Core Functions with explicit docstrings
def get_summary(user_input: str) -> str:
    """Generate a brief summary of the user's input.
   
    Args:
        user_input (str): The user's description of their experience.
   
    Returns:
        str: A summary string prefixed with 'User is experiencing:'.
    """
    logger.debug(f"[WELLNESS_CHECK_AGENT] Summarizing user's concern: {user_input}")
    return f"User is experiencing: {user_input}"

def analyze_sentiment(summary: str) -> str:
    """Perform basic sentiment analysis and return an empathetic response.
   
    Args:
        summary (str): The summary of the user's input.
   
    Returns:
        str: An empathetic response based on detected sentiment.
    """
    logger.debug("[WELLNESS_CHECK_AGENT] Analyzing sentiment...")
    if "hopeless" in summary.lower() or "overwhelmed" in summary.lower():
        return "I'm really sorry you're feeling this way. You're not alone, and support is available."
    elif "stressed" in summary.lower() or "anxious" in summary.lower():
        return "I understand that stress can be overwhelming. Taking small steps can really help."
    return "It's okay to have tough days. Remember, you're stronger than you think!"

def generate_questions(summary: str) -> list:
    """Generate context-based questions based on the user's summary.
   
    Args:
        summary (str): The summary of the user's input.
   
    Returns:
        list: A list of five questions tailored to the summary.
    """
    logger.debug("[WELLNESS_CHECK_AGENT] Generating dynamic questions...")
    return [
        f"Can you describe more about how {summary.split(':')[-1].strip()} affects your daily life?",
        "Do you experience mood swings frequently? (Yes/No)",
        "Have you had trouble sleeping lately? (Yes/No)",
        "Do you feel disconnected from loved ones? (Yes/No)",
        "On a scale of 1-10, how would you rate your emotional well-being?"
    ]

def wellness_check(patient_mood: str, additional_responses: list = None) -> str:
    """Assess the patient's mental health based on their mood description.
   
    Args:
        patient_mood (str): The user's description of their mood.
        additional_responses (list, optional): Responses to follow-up questions. Defaults to None.
   
    Returns:
        str: A risk assessment or questions if more info is needed.
    """
    logger.info(f"[WELLNESS_CHECK_AGENT] Assessing mood: {patient_mood}")
    summary = get_summary(patient_mood)
    sentiment_response = analyze_sentiment(summary)
   
    if additional_responses is None or len(additional_responses) < 5:
        questions = generate_questions(summary)
        return f"{sentiment_response} Would you be comfortable answering these questions? {questions}"
   
    responses = [r.lower() for r in additional_responses]
    if "yes" in responses or any(int(x) >= 7 for x in responses if x.isdigit()):
        return "High risk detected. Our therapist will contact you soon."
    elif any(int(x) >= 4 for x in responses if x.isdigit()):
        return "Medium risk detected. Remember, tough times don't last. You are stronger than you think!"
    else:
        return "Low risk detected. Keep going! Every storm runs out of rain."

def provide_therapy(condition: str, user_id: str = None, chat_history: list = None) -> dict:
    """Generate a personalized therapy recommendation and plan.
   
    Args:
        condition (str): The user's reported condition or need for therapy.
        user_id (str, optional): The user's ID. Defaults to None.
        chat_history (list, optional): The user's chat history. Defaults to None.
   
    Returns:
        dict: Contains 'user_response' for the user and 'therapy_plan' for the therapist.
    """
    logger.info(f"[THERAPY_AGENT] Recommending therapy for: {condition}")
   
    condition_lower = condition.lower()
    chat_summary = "No chat history provided."
    if chat_history:
        user_msgs = [msg["content"] for msg in chat_history if msg["role"] == "user"]
        chat_summary = "Chat history: " + "; ".join(user_msgs[-3:])
   
    if "anxiety" in condition_lower:
        therapy_type = "Cognitive Behavioral Therapy (CBT)"
        plan = {
            "short_term": "Practice deep breathing exercises daily for 10 minutes.",
            "long_term": "Develop a structured routine to reduce uncertainty.",
            "coping": "Use grounding techniques (5-4-3-2-1 method) during anxious moments."
        }
    elif "depression" in condition_lower:
        therapy_type = "Mindfulness-Based Therapy"
        plan = {
            "short_term": "Engage in 5 minutes of mindfulness meditation daily.",
            "long_term": "Build a support network and set small achievable goals.",
            "coping": "Journal thoughts to identify negative patterns."
        }
    else:
        therapy_type = "General Therapy"
        plan = {
            "short_term": "Reflect on current feelings for 5 minutes daily.",
            "long_term": "Explore underlying causes with a professional.",
            "coping": "Reach out to a friend or family member when needed."
        }

    therapy_plan_text = (
        f"Patient Condition: {condition}\n"
        f"Chat Summary: {chat_summary}\n"
        f"Recommended Therapy: {therapy_type}\n"
        f"Therapy Plan:\n"
        f"  - Short-term: {plan['short_term']}\n"
        f"  - Long-term: {plan['long_term']}\n"
        f"  - Coping Mechanism: {plan['coping']}\n"
        f"Generated on: {datetime.now(timezone.utc).isoformat()}\n"
        f"User ID: {user_id or 'Not provided'}"
    )

    user_response = (
        f"Recommended therapy: {therapy_type}. "
        "I've prepared a personalized therapy plan and sent it to your therapist for further evaluation. "
        "They’ll reach out to you soon!"
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
    logger.info(f"[POST_REHAB_AGENT] Following up: {patient_status}")
    
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
        logger.error("Missing SMTP configuration in environment variables")
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
        logger.info(f"Therapy plan emailed to {therapist_email} for user_id={user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email for user_id={user_id}: {str(e)}")
        return False

# Log agent creation
logger.info("Creating agents...")
try:
    wellness_check_agent = create_react_agent(
        model=model,
        tools=[wellness_check],
        name="wellness_check_expert",
        prompt="You are an expert in mental health assessments. For any user message describing their mood or emotions, use the 'wellness_check' tool to assess their mental state. Pass the user's message as the 'patient_mood' parameter and, if provided, any additional responses as 'additional_responses'. Return the tool's output as your response."
    )
    logger.info("Wellness check agent created successfully")
except Exception as e:
    logger.error(f"Failed to create wellness_check_agent: {str(e)}")
    raise

try:
    therapy_agent = create_react_agent(
        model=model,
        tools=[provide_therapy,send_therapy_plan_to_therapist],
        name="therapy_expert",
        prompt="You are the Personalized Therapy Agent, part of SoulSync AI. When a user requests therapy or indicates a need for professional support, use the 'provide_therapy' tool with the user's message as the 'condition' parameter, their user_id, and chat history. Return the 'user_response' from the tool’s output as your response. The tool will handle sending the therapy plan to the therapist."
    )
    logger.info("Therapy agent created successfully")
except Exception as e:
    logger.error(f"Failed to create therapy_agent: {str(e)}")
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

    logger.info("Post-rehab agent created successfully")
except Exception as e:
    logger.error(f"Failed to create post_rehab_agent: {str(e)}")
    raise

# Create Supervisor Agent
logger.info("Creating supervisor agent...")
try:
    chat_agent = create_supervisor(
        [wellness_check_agent, therapy_agent, post_rehab_agent],
        model=model,
        prompt=(
            "Your role is to:\n"
            "1. Understand the user's needs and emotions based on their message and conversation history.\n"
            "2. Provide an empathetic, natural response without mentioning the routing process.\n"
           
            "3. Classify the user's intent and route to the appropriate agent with necessary data:\n"
            "   - Intent: wellness (emotional distress phrases like 'I feel sad', 'I’m stressed') -> Route to wellness_check_agent with patient_mood=user_message and additional_responses if provided\n"
            "   - Intent: therapy (explicit requests like 'I need therapy', 'I want a therapist') -> Route to therapy_agent with condition=user_message, user_id, and chat_history\n"
            "   - Intent: rehab (post-rehab phrases like 'I relapsed', 'I'm struggling after rehab', 'My therapy is finished but I have problems') -> Route to post_rehab_agent with patient_status=user_message\n"
            "   - Intent: general (other messages like 'hi', 'what is this') -> Respond empathetically without routing, e.g., 'Hello! I’m here to help—how are you feeling today?'\n"
           "**Instructions**:\n"
            "- Prioritize explicit keywords: 'therapy' -> therapy intent; emotional distress -> wellness intent.\n"
            "- Use conversation history to determine emotional state.\n"
            "- Do **not** modify agent responses; present them exactly as they are."
        )
    )
    logger.info("Supervisor agent created successfully")
except Exception as e:
    logger.error(f"Failed to create chat_agent: {str(e)}")
    raise

# Compile the workflow
logger.info("Compiling workflow...")
app_ai = chat_agent.compile()
logger.info("Workflow compiled successfully")

# Cosmos DB Helper Functions
def get_user_data(user_id: str) -> dict:
    """Retrieve user data from Cosmos DB."""
    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        return user_docs[0] if user_docs else {}
    except Exception as e:
        logger.error(f"get_user_data: Error for user_id={user_id}: {str(e)}")
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
        logger.debug(f"Stored user data for user_id={user_id}")
    except Exception as e:
        logger.error(f"store_user_data: Error for user_id={user_id}: {str(e)}")
        raise

# API Routes
@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    required_fields = ["firstName", "lastName", "dob", "email", "password"]
   
    if not all(field in data for field in required_fields):
        logger.warning("Signup failed: Missing required fields")
        return jsonify({"error": "Missing required fields"}), 400

    query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
    existing_users = list(container.query_items(query, enable_cross_partition_query=True))
    if existing_users:
        logger.warning(f"Signup failed: Email {data['email']} already exists")
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
    logger.info(f"User signed up successfully: {user_id}")
    return jsonify({"message": "Signup successful", "user_id": user_id}), 201

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data.get("email") or not data.get("password"):
        logger.warning("Login failed: Missing email or password")
        return jsonify({"error": "Missing email or password"}), 400

    query = f"SELECT * FROM c WHERE c.email = '{data['email']}'"
    users = list(container.query_items(query, enable_cross_partition_query=True))
   
    if not users or users[0]["password"] != data["password"]:
        logger.warning(f"Login failed for email {data['email']}: Invalid credentials")
        return jsonify({"error": "Invalid email or password"}), 401

    logger.info(f"User logged in: {users[0]['user_id']}")
    return jsonify({"message": "Login successful", "user_id": users[0]["user_id"]}), 200

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history():
    user_id = request.args.get("user_id")
    if not user_id:
        logger.warning("Missing user_id in get_chat_history request")
        return jsonify({"error": "Missing user_id"}), 400

    try:
        logger.debug(f"Fetching chat history for user_id={user_id}")
        user_data = get_user_data(user_id)
        if not user_data:
            logger.warning(f"No user data found for user_id={user_id}")
            return jsonify({"messages": []}), 200
        messages = user_data.get("messages", [])
        logger.info(f"Chat history retrieved for user_id={user_id}: {len(messages)} messages")
        return jsonify({"messages": messages})
    except Exception as e:
        logger.exception(f"Error fetching chat history for user_id={user_id}: {str(e)}")
        return jsonify({"error": "Failed to fetch chat history"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", "")
    additional_responses = data.get("additional_responses", [])
   
    if not user_message or not user_id:
        logger.warning("Chat request failed: Missing user_id or message")
        return jsonify({"error": "Missing user_id or message"}), 400
   
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            logger.warning(f"No user data found for user_id={user_id}")
            return jsonify({"error": "User not found"}), 404

        messages = user_data.get("messages", [])
        context = user_data.get("context", {})
       
        messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        logger.debug(f"User message added for user_id={user_id}: {user_message}")

        last_user_messages = [msg["content"] for msg in messages[-3:] if msg["role"] == "user"]
        if len(last_user_messages) >= 2 and all(msg == last_user_messages[-1] for msg in last_user_messages[-2:]):
            response = "I notice you've said the same thing a few times. Can you tell me more about how you’re feeling?"
            agent_used = "chat_agent"
            logger.info(f"Repetitive message detected for user_id={user_id}, handled by chat_agent")
        else:
            langchain_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages[-5:]]
            response_data = {
                "messages": langchain_messages,
                "additional_responses": additional_responses,
                "user_id": user_id,
                "chat_history": messages
            }
            logger.debug(f"Input to supervisor for user_id={user_id}: {response_data}")
           
            max_iterations = 10
            iteration = 0
            final_response = None
            agent_used = "chat_agent"  # Default

            while iteration < max_iterations:
                response_data = app_ai.invoke(response_data)
                iteration += 1
                next_agent = response_data.get("next_agent")
                logger.info(f"Supervisor iteration {iteration} for user_id={user_id}: Routing to {next_agent if next_agent else 'no further agent'}")

                # Check the last message for tool output or agent response
                last_message = response_data["messages"][-1]
                logger.debug(f"Last message in iteration {iteration}: {last_message}")

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
                logger.warning(f"Max iterations reached for user_id={user_id}, defaulting to chat_agent")

            response = final_response
            logger.info(f"Final response for user_id={user_id} from {agent_used}: {response}")

            # Handle therapy agent email sending
            if agent_used == "therapy_expert" and isinstance(last_message, dict) and "therapy_plan" in last_message:
                success = send_therapy_plan_to_therapist(last_message["therapy_plan"], user_id)
                if not success:
                    response += " (Note: There was an issue sending the plan to your therapist; we’ll retry later.)"
                logger.info(f"Therapy plan email attempt for user_id={user_id}: {'Success' if success else 'Failed'}")

        messages.append({
            "role": "assistant",
            "content": response,
            "agent_name": agent_used,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        store_user_data(user_id, messages, context)
        return jsonify({"user_id": user_id, "response": response, "agent_used": agent_used})
    except Exception as e:
        logger.exception(f"Chat Error for user_id={user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/welcome", methods=["POST"])
def welcome():
    logger.info("Welcome endpoint called")
    return jsonify({"response": "Welcome to SoulSync! How can I assist you today?"})

if __name__ == "__main__":
    logger.info("Starting SoulSync application...")
    app.run(debug=True, host="127.0.0.1", port=5000)








