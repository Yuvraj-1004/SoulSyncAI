import os
from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

# ========== Define Agent Functionalities ========== #

def get_summary(user_input: str) -> str:
    """Generate a brief summary of the user's experience."""
    print(f"[WELLNESS_CHECK_AGENT] Summarizing user's concern: {user_input}")
    return f"User is experiencing: {user_input}"


def analyze_sentiment(summary: str) -> str:
    """Perform sentiment analysis and provide an empathetic response."""
    print("[WELLNESS_CHECK_AGENT] Analyzing sentiment...")
    if "hopeless" in summary or "overwhelmed" in summary:
        return "I'm really sorry you're feeling this way. You're not alone, and support is available."
    elif "stressed" in summary or "anxious" in summary:
        return "I understand that stress can be overwhelming. Taking small steps can really help."
    return "It's okay to have tough days. Remember, you're stronger than you think!"


def generate_questions(summary: str) -> list:
    """Generate five context-based questions if the summary is insufficient."""
    print("[WELLNESS_CHECK_AGENT] Generating dynamic questions...")
    return [
        f"Can you describe more about how {summary.split(':')[-1]} affects your daily life?",
        "Do you experience mood swings frequently? (Yes/No)",
        "Have you had trouble sleeping lately? (Yes/No)",
        "Do you feel disconnected from loved ones? (Yes/No)",
        "On a scale of 1-10, how would you rate your emotional well-being?"
    ]


def wellness_check(patient_mood: str) -> str:
    """Assess the patient's mental health based on mood description."""
    print(f"[WELLNESS_CHECK_AGENT] Assessing mood: {patient_mood}")
    summary = get_summary(patient_mood)
    sentiment_response = analyze_sentiment(summary)
    user_agree = input(f"[WELLNESS_CHECK_AGENT] {sentiment_response} Would you be comfortable answering a few more questions? (Yes/No)\nYou: ")
    
    if user_agree.lower() == "yes":
        questions = generate_questions(summary)
        responses = []
        for question in questions:
            user_response = input(f"[WELLNESS_CHECK_AGENT] {question}\nYou: ")
            responses.append(user_response.lower())
        
        if "yes" in responses or any(int(x) >= 7 for x in responses if x.isdigit()):
            return "High risk detected. Our therapist will contact you soon."
        elif any(int(x) >= 4 for x in responses if x.isdigit()):
            return "Medium risk detected. Remember, tough times don't last. You are stronger than you think!"
        else:
            return "Low risk detected. Keep going! Every storm runs out of rain."
    
    return "Thank you for sharing. If you ever need support, don't hesitate to reach out."


def provide_therapy(condition: str) -> str:
    """Recommend therapy based on the patient's condition."""
    print(f"[THERAPY_AGENT] Recommending therapy for: {condition}")
    if "anxiety" in condition.lower():
        return "Recommended therapy: Cognitive Behavioral Therapy (CBT)."
    elif "depression" in condition.lower():
        return "Recommended therapy: Mindfulness-Based Therapy."
    return "General therapy session is advised."


def post_rehab_followup(patient_status: str) -> str:
    """Engage with patient after therapy completion."""
    print(f"[POST_REHAB_AGENT] Following up: {patient_status}")
    if "improved" in patient_status.lower():
        return "Continue with self-help practices and regular check-ins."
    return "Schedule a follow-up therapy session."

# ========== Create Agents ========== #

wellness_check_agent = create_react_agent(
    model=model,
    tools=[wellness_check],
    name="wellness_check_expert",
    prompt="You are an expert in mental health assessments. Gather a summary of user experience, analyze sentiment, and if necessary, generate 5 personalized questions before determining risk."
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

# ========== Create Supervisor Agent ========== #

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

# ========== Interactive Chat Mode ========== #

def chat_with_agents():
    print("\nWelcome to the AI Healthcare Assistant! Type 'exit' to end the chat.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        print("\n" + "="*50)
        print(f"[TEST] User Query: {user_query}")
        print("="*50)
        
        try:
            response = app.invoke({"messages": [{"role": "user", "content": user_query}]})
            messages = response.get("messages", [])
            formatted_response = "\n[SYSTEM RESPONSE]\n"
            
            for message in messages:
                agent_name = getattr(message, "name", "Unknown Agent")
                content = getattr(message, "content", "No response.")
                formatted_response += f"[{agent_name}]: {content}\n"
            
            print(formatted_response)
        except Exception as e:
            print("[ERROR]:", str(e))

# Start interactive chat
chat_with_agents()
