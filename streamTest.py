import os
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Streamlit app
st.set_page_config(page_title="AI Healthcare Assistant", layout="wide")
st.title("AI Healthcare Assistant 💬")

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

# Define Agent Functionalities
def get_summary(user_input: str) -> str:
    """Generate a brief summary of the user's experience."""
    return f"User is experiencing: {user_input}"

def analyze_sentiment(summary: str) -> str:
    """"Hello"""
    if "hopeless" in summary or "overwhelmed" in summary:
        return "I'm really sorry you're feeling this way. You're not alone, and support is available."
    elif "stressed" in summary or "anxious" in summary:
        return "I understand that stress can be overwhelming. Taking small steps can really help."
    return "It's okay to have tough days. Remember, you're stronger than you think!"

def generate_questions(summary: str) -> list:
    """"Hello"""
    return [
        f"Can you describe more about how {summary.split(':')[-1]} affects your daily life?",
        "Do you experience mood swings frequently? (Yes/No)",
        "Have you had trouble sleeping lately? (Yes/No)",
        "Do you feel disconnected from loved ones? (Yes/No)",
        "On a scale of 1-10, how would you rate your emotional well-being?"
    ]

def wellness_check(patient_mood: str) -> str:
    """"Hello"""
    summary = get_summary(patient_mood)
    sentiment_response = analyze_sentiment(summary)
    return sentiment_response

def provide_therapy(condition: str) -> str:
    """"Hello"""
    if "anxiety" in condition.lower():
        return "Recommended therapy: Cognitive Behavioral Therapy (CBT)."
    elif "depression" in condition.lower():
        return "Recommended therapy: Mindfulness-Based Therapy."
    return "General therapy session is advised."

def post_rehab_followup(patient_status: str) -> str:
    """"Hello"""
    if "improved" in patient_status.lower():
        return "Continue with self-help practices and regular check-ins."
    return "Schedule a follow-up therapy session."

# Create Agents
wellness_check_agent = create_react_agent(
    model=model,
    tools=[wellness_check],
    name="wellness_check_expert",
    prompt="You are an expert in mental health assessments. Gather a summary of user experience, analyze sentiment, and generate a response."
)

therapy_agent = create_react_agent(
    model=model,
    tools=[provide_therapy],
    name="therapy_expert",
    prompt="You are a therapy expert. Recommend the best therapy based on the patient's condition."
)

post_rehab_agent = create_react_agent(
    model=model,
    tools=[post_rehab_followup],
    name="post_rehab_expert",
    prompt="You help patients post-therapy. Engage with them and suggest the next steps."
)

# Create Supervisor Agent
chat_agent = create_supervisor(
    [wellness_check_agent, therapy_agent, post_rehab_agent],
    model=model,
    prompt=(
        "You are the Chat Agent. Route queries based on the patient's needs: "
        "- If it's about mental health assessment, use wellness_check_agent. "
        "- If therapy recommendations are needed, use therapy_agent. "
        "- If it's a follow-up after therapy, use post_rehab_agent. "
        "If the user is in distress, let wellness_check_agent take over control."
    )
)

# Compile the workflow
app = chat_agent.compile()

# Streamlit Chat UI
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for role, text in st.session_state.chat_history:
    role="assistant"
    with st.chat_message(role):
        st.write(text)

# User input
user_query = st.chat_input("Type your message...")
if user_query:
    st.session_state.chat_history.append(("user", user_query))
    
    response = app.invoke({"messages": [{"role": "user", "content": user_query}]})
    messages = response.get("messages", [])
    
    for message in messages:
        agent_name = getattr(message, "name", "assistant")
        content = getattr(message, "content", "No response.")
        st.session_state.chat_history.append((agent_name, content))
    
    st.rerun()
