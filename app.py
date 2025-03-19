# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from openai import AzureOpenAI
# from azure.cosmos import CosmosClient, exceptions
# from pathlib import Path
# from dotenv import load_dotenv
# import uuid
# from typing import TypedDict, List
# from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.memory import InMemorySaver

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

# # Signup endpoint (unchanged)
# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.json
#     first_name = data.get("firstName")
#     last_name = data.get("lastName")
#     dob = data.get("dob")
#     email = data.get("email")
#     password = data.get("password")

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
#         "password": password,
#         "conversation_history": [],
#         "baseline_emotional_state": "neutral"
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
#         if user_doc["password"] != password:
#             return jsonify({"error": "Incorrect password"}), 401

#         return jsonify({"user_id": user_doc["user_id"], "message": "Login successful"}), 200

#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

# def check_wellness(status: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Assess mood of user by continuing Asking questions to user analyzing their sentiment. If sentiment unclear, then ask them to give answer to 5 pyshchometric questions one by one, and record the response and determine the risk level if the risk is high then ask them to talk to a human, else send them motivational messages. If still unclear ask them a set of last 5 questions again. Questions should be randomly generated based on the flow of chat."}, {"role": "user", "content": status}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return "Sorry, I couldn’t assess your wellness right now."
    
# # Define tools for specialized agents (unchanged)
# def suggest_therapy_plan(input_text: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": (
#                 "You are an expert in creating personalized, readable therapy plans. Respond with a warm, supportive tone."
#             )}, {"role": "user", "content": input_text}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return "Sorry, I couldn’t suggest a therapy plan right now."

# def post_rehab_followup(input_text: str) -> str:
#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "system", "content": "Provide post-rehab follow-up advice."}, {"role": "user", "content": input_text}]
#         )
#         return response.choices[0].message.content.strip()
#     except Exception as e:
#         return "Sorry, I couldn’t provide post-rehab advice right now."

# # Custom AzureChatOpenAI wrapper (unchanged)
# class AzureChatOpenAI:
#     def __init__(self, client):
#         self.client = client

#     def invoke(self, messages):
#         try:
#             response = self.client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=messages
#             )
#             return {"role": "assistant", "content": response.choices[0].message.content.strip()}
#         except Exception as e:
#             raise

# # Define state schema with user_name added
# class AgentState(TypedDict):
#     messages: List[dict]
#     next: str
#     selected_agent: str
#     topics: List[str]
#     sentiment: str
#     user_name: str  # Added for personalized greetings

# # Chat Agent (Updated to send greeting)
# def create_chat_agent(model):
#     def chat_node(state: AgentState) -> AgentState:
#         messages = state["messages"]
#         user_name = state.get("user_name", "friend")  # Default to "friend" if no name

#         # Check if this is the start of the chat (no user messages yet)
#         user_messages = [m for m in messages if m["role"] == "user"]
#         is_start_of_chat = not user_messages

#         if is_start_of_chat:
#             # Determine if user is new or existing
#             is_new_user = len(messages) == 0
#             if is_new_user:
#                 greeting = (
#                     f"Hi there, {user_name}! I'm SoulSync, your friendly AI assistant. "
#                     "I’m here to support you on your mental health journey. What’s on your mind today?"
#                 )
#             else:
#                 greeting = (
#                     f"Welcome back, {user_name}! Great to see you again. "
#                     "How have you been since our last chat?"
#                 )
#             updated_messages = messages + [{"role": "assistant", "content": greeting, "agent": "chat_agent"}]
#             return {
#                 "messages": updated_messages,
#                 "topics": [],
#                 "sentiment": "neutral",
#                 "next": "supervisor",
#                 "selected_agent": "chat_agent"
#             }

#         # Process user message if present
#         last_message = messages[-1]["content"]
#         empathetic_prompt = [
#             {"role": "system", "content": (
#                 "You are a warm, empathetic conversational partner. Respond to the user's message in a supportive way. "
#                 "Extract main topics and sentiment."
#             )}
#         ] + messages
#         try:
#             empathetic_response = model.invoke(empathetic_prompt)
#             empathetic_message = empathetic_response["content"]
#         except Exception as e:
#             empathetic_message = "I’m here for you! Let’s talk about how you’re feeling."

#         # Extract topics and sentiment (unchanged)
#         analysis_prompt = [
#             {"role": "system", "content": (
#                 "Analyze the user   's message and extract the main topics and sentiment. "
#                 "Return: Topics: [topic1, topic2], Sentiment: sentiment"
#             )}
#         ] + messages
#         try:
#             analysis_response = model.invoke(analysis_prompt)
#             analysis_result = analysis_response["content"]
#             topics = analysis_result.split("Topics: ")[1].split(", Sentiment: ")[0].strip("[]").split(", ")
#             sentiment = analysis_result.split("Sentiment: ")[1].strip()
#         except Exception as e:
#             topics = ["unknown"]
#             sentiment = "neutral"

#         updated_messages = messages + [{"role": "assistant", "content": empathetic_message, "agent": "chat_agent"}]
#         return {
#             "messages": updated_messages,
#             "topics": topics,
#             "sentiment": sentiment,
#             "next": "supervisor",
#             "selected_agent": "chat_agent"
#         }

#     graph = StateGraph(AgentState)
#     graph.add_node("chat_agent", chat_node)
#     graph.set_entry_point("chat_agent")
#     return graph.compile(name="chat_agent")

# # Custom Agent (unchanged)
# def create_custom_agent(model, tool, name, prompt):
#     def agent_node(state: AgentState) -> AgentState:
#         messages = state["messages"]
#         topics = state["topics"]
#         sentiment = state["sentiment"]
#         system_prompt = [{"role": "system", "content": prompt}]
#         tool_prompt = [
#             {"role": "system", "content": (
#                 f"Process the user's input using the tool '{tool.__name__}'. "
#                 f"User input: {messages[-1]['content']}, Topics: {topics}, Sentiment: {sentiment}"
#             )}
#         ]
#         try:
#             response = model.invoke(system_prompt + messages + tool_prompt)
#             tool_result = tool(messages[-1]["content"])
#             return {
#                 "messages": messages + [{"role": "assistant", "content": tool_result, "agent": name}],
#                 "selected_agent": name
#             }
#         except Exception as e:
#             return {
#                 "messages": messages + [{"role": "assistant", "content": f"Sorry, something went wrong with {name}."}],
#                 "selected_agent": name
#             }

#     graph = StateGraph(AgentState)
#     graph.add_node(name, agent_node)
#     graph.set_entry_point(name)
#     graph.add_edge(name, END)
#     return graph.compile(name=name)

# # Supervisor (unchanged)
# def create_custom_supervisor(agents, model, prompt, checkpointer):
#     def supervisor_node(state: AgentState) -> AgentState:
#         messages = state["messages"]
#         topics = state["topics"]
#         sentiment = state["sentiment"]
#         system_prompt = [{"role": "system", "content": prompt}]
#         routing_prompt = system_prompt + messages + [
#             {"role": "system", "content": (
#                 "Route to: 'personalized_plan_update_agent', 'wellness_check_agent', or 'post_rehab_follow_up_agent'. "
#                 "Return only the agent name based on topics and sentiment."
#             )}
#         ]
#         try:
#             response = model.invoke(routing_prompt)
#             next_agent = response["content"].strip()
#         except Exception as e:
#             return {"messages": messages + [{"role": "assistant", "content": "Sorry, I couldn’t decide."}], "selected_agent": "supervisor"}

#         if next_agent in ["personalized_plan_update_agent", "wellness_check_agent", "post_rehab_follow_up_agent"]:
#             return {"next": next_agent, "messages": messages, "selected_agent": next_agent}
#         return {"messages": messages + [{"role": "assistant", "content": "Sorry, couldn’t route."}], "selected_agent": "supervisor"}

#     graph = StateGraph(AgentState)
#     graph.add_node("chat_agent", agents[0])
#     graph.add_node("supervisor", supervisor_node)
#     for agent in agents[1:]:
#         graph.add_node(agent.name, agent)
#     graph.set_entry_point("chat_agent")
#     graph.add_edge("chat_agent", "supervisor")
#     graph.add_conditional_edges("supervisor", lambda state: state.get("next", END), {
#         "personalized_plan_update_agent": "personalized_plan_update_agent",
#         "wellness_check_agent": "wellness_check_agent",
#         "post_rehab_follow_up_agent": "post_rehab_follow_up_agent",
#         END: END
#     })
#     for agent in agents[1:]:
#         graph.add_edge(agent.name, END)
#     return graph.compile(checkpointer=checkpointer)

# # Initialize Supervisor (unchanged except for user_name in state)
# def initialize_supervisor():
#     model = AzureChatOpenAI(client)
#     chat_agent = create_chat_agent(model)
#     personalized_plan_update_agent = create_custom_agent(
#         model=model, tool=suggest_therapy_plan, name="personalized_plan_update_agent",
#         prompt="You are an expert in creating therapy plans."
#     )
#     wellness_check_agent = create_custom_agent(
#         model=model, tool=check_wellness, name="wellness_check_agent",
#         prompt="You are a wellness monitoring expert."
#     )
#     post_rehab_follow_up_agent = create_custom_agent(
#         model=model, tool=post_rehab_followup, name="post_rehab_follow_up_agent",
#         prompt="You are a post-rehab follow-up expert."
#     )
#     agents = [chat_agent, personalized_plan_update_agent, wellness_check_agent, post_rehab_follow_up_agent]
#     checkpointer = InMemorySaver()
#     workflow = create_custom_supervisor(
#         agents=agents, model=model,
#         prompt="You are SoulSync, a supervisor routing messages based on topics and sentiment.",
#         checkpointer=checkpointer
#     )
#     return workflow

# # Global supervisor instance
# app.supervisor = initialize_supervisor()

# # Updated Chat Endpoint
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.json
#     user_id = data.get("user_id")
#     user_message = data.get("message", "").strip()  # Empty message triggers greeting

#     if not user_id:
#         return jsonify({"error": "Missing user_id"}), 400

#     try:
#         query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
#         user_docs = list(container.query_items(query, enable_cross_partition_query=True))
#         if not user_docs:
#             return jsonify({"error": "User not found"}), 404

#         user_doc = user_docs[0]
#         conversation_history = user_doc.get("conversation_history", [])
#         user_name = user_doc.get("first_name", "friend")

#         # Prepare initial messages: empty for greeting, or include user message
#         if not user_message:
#             initial_messages = conversation_history
#         else:
#             initial_messages = conversation_history + [{"role": "user", "content": user_message}]

#         # Invoke the supervisor with user_name included
#         result = app.supervisor.invoke(
#             {
#                 "messages": initial_messages,
#                 "user_name": user_name,
#                 "topics": [],
#                 "sentiment": "neutral"
#             },
#             config={"configurable": {"thread_id": user_id}}
#         )

#         # Extract response and agent
#         last_message = result["messages"][-1]
#         ai_response = last_message["content"]
#         selected_agent = last_message.get("agent", "chat_agent")

#         # Update conversation history
#         if not user_message:
#             conversation_history.append({"role": "assistant", "content": ai_response, "agent": selected_agent})
#         else:
#             conversation_history.append({"role": "user", "content": user_message})
#             conversation_history.append({"role": "assistant", "content": ai_response, "agent": selected_agent})
#         user_doc["conversation_history"] = conversation_history
#         container.upsert_item(user_doc)

#         return jsonify({"response": ai_response, "agent": selected_agent})

#     except exceptions.CosmosHttpResponseError as e:
#         return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
#     except Exception as e:
#         return jsonify({"error": f"Error: {str(e)}"}), 500

# @app.route("/")
# def home():
#     return "SoulSync API is running!"

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
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

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

# Signup endpoint (unchanged)
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    first_name = data.get("firstName")
    last_name = data.get("lastName")
    dob = data.get("dob")
    email = data.get("email")
    password = data.get("password")

    if not all([first_name, last_name, dob, email, password]):
        return jsonify({"error": "Missing required fields"}), 400

    user_id = str(uuid.uuid4())
    user_data = {
        "id": user_id,
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "email": email,
        "password": password,
        "conversation_history": [],
        "baseline_emotional_state": "neutral"
    }

    try:
        container.upsert_item(user_data)
        return jsonify({"user_id": user_id, "message": "Signup successful"}), 201
    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

# Login endpoint (unchanged)
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.email = '{email}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))

        if not user_docs:
            return jsonify({"error": "Email not found"}), 404

        user_doc = user_docs[0]
        if user_doc["password"] != password:
            return jsonify({"error": "Incorrect password"}), 401

        return jsonify({"user_id": user_doc["user_id"], "message": "Login successful"}), 200

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500

# Define tools for specialized agents (unchanged)
def suggest_therapy_plan(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": (
                "You are an expert in creating personalized, readable therapy plans. Respond with a warm, supportive tone."
            )}, {"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn’t suggest a therapy plan right now."

def check_wellness(status: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Assess mood via chatting then motivate the patient, if unclear via chat provide them with a set of questions to answer to check the mood "}, {"role": "user", "content": status}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn’t assess your wellness right now."

def post_rehab_followup(input_text: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Provide post-rehab follow-up advice."}, {"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Sorry, I couldn’t provide post-rehab advice right now."

# Custom AzureChatOpenAI wrapper (unchanged)
class AzureChatOpenAI:
    def __init__(self, client):
        self.client = client

    def invoke(self, messages):
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return {"role": "assistant", "content": response.choices[0].message.content.strip()}
        except Exception as e:
            raise

# Define state schema
class AgentState(TypedDict):
    messages: List[dict]
    selected_agent: str
    topics: List[str]
    sentiment: str
    user_name: str

# Chat Agent (Now acts as the supervisor)
def create_chat_agent(model, sub_agents):
    def chat_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        user_name = state.get("user_name", "friend")

        # Check if this is the start of the chat (no user messages yet)
        user_messages = [m for m in messages if m["role"] == "user"]
        is_start_of_chat = not user_messages

        if is_start_of_chat:
            # Determine if user is new or existing
            is_new_user = len(messages) == 0
            if is_new_user:
                greeting = (
                    f"Hi there, {user_name}! 👋 I'm SoulSync, your friendly AI assistant. "
                    "I’m here to support you on your mental health journey. What’s on your mind today?"
                )
            else:
                greeting = (
                    f"Welcome back, {user_name}! 😊 Great to see you again. "
                    "How have you been since our last chat?"
                )
            updated_messages = messages + [{"role": "assistant", "content": greeting, "agent": "chat_agent"}]
            return {
                "messages": updated_messages,
                "topics": [],
                "sentiment": "neutral",
                "selected_agent": "chat_agent"
            }

        # Process user message
        last_message = messages[-1]["content"]

        # Extract topics and sentiment
        analysis_prompt = [
            {"role": "system", "content": (
                "Analyze the user's message and extract the main topics and sentiment. "
                "Return: Topics: [topic1, topic2], Sentiment: sentiment"
            )}
        ] + messages
        try:
            analysis_response = model.invoke(analysis_prompt)
            analysis_result = analysis_response["content"]
            topics = analysis_result.split("Topics: ")[1].split(", Sentiment: ")[0].strip("[]").split(", ")
            sentiment = analysis_result.split("Sentiment: ")[1].strip()
        except Exception as e:
            topics = ["unknown"]
            sentiment = "neutral"

        # Routing logic (previously in supervisor)
        routing_prompt = [
            {"role": "system", "content": (
                "You are SoulSync, a chill AI assistant. Based on the conversation history, extracted topics, and sentiment, decide which agent to route to: "
                "'personalized_plan_update_agent', 'wellness_check_agent', or 'post_rehab_follow_up_agent'. "
                "Return only the agent name. Use the following guidelines:\n"
                "- 'personalized_plan_update_agent' for requests related to therapy suggestions\n"
                "- 'wellness_check_agent' for doing sentiment analysis and detecting mood,.\n"
                "- 'post_rehab_follow_up_agent' for follow-up after rehab.\n"
                f"Current topics: {topics}, Sentiment: {sentiment}. Consider the intent and context."
            )}
        ] + messages
        try:
            response = model.invoke(routing_prompt)
            next_agent = response["content"].strip()
        except Exception as e:
            next_agent = "chat_agent"  # Default to chat_agent if routing fails

        # If no specialized agent is selected, respond empathetically
        if next_agent == "chat_agent" or next_agent not in sub_agents:
            empathetic_prompt = [
                {"role": "system", "content": (
                    "You are a warm, empathetic conversational partner. Respond to the user's message in a supportive way and do not answer the queries that are out of the ."
                )}
            ] + messages
            try:
                empathetic_response = model.invoke(empathetic_prompt)
                empathetic_message = empathetic_response["content"]
            except Exception as e:
                empathetic_message = "I’m here for you! Let’s talk about how you’re feeling."
            updated_messages = messages + [{"role": "assistant", "content": empathetic_message, "agent": "chat_agent"}]
            return {
                "messages": updated_messages,
                "topics": topics,
                "sentiment": sentiment,
                "selected_agent": "chat_agent"
            }

        # Route to the selected sub-agent
        sub_agent = sub_agents[next_agent]
        try:
            sub_agent_result = sub_agent.invoke({
                "messages": messages,
                "topics": topics,
                "sentiment": sentiment,
                "user_name": user_name
            })
            updated_messages = sub_agent_result["messages"]
            selected_agent = sub_agent_result["selected_agent"]
        except Exception as e:
            updated_messages = messages + [{"role": "assistant", "content": "Sorry, I couldn’t process that. Let’s try again!", "agent": "chat_agent"}]
            selected_agent = "chat_agent"

        return {
            "messages": updated_messages,
            "topics": topics,
            "sentiment": sentiment,
            "selected_agent": selected_agent
        }

    graph = StateGraph(AgentState)
    graph.add_node("chat_agent", chat_node)
    graph.set_entry_point("chat_agent")
    graph.add_edge("chat_agent", END)
    return graph.compile(name="chat_agent")

# Custom Agent (unchanged)
def create_custom_agent(model, tool, name, prompt):
    def agent_node(state: AgentState) -> AgentState:
        messages = state["messages"]
        topics = state["topics"]
        sentiment = state["sentiment"]
        system_prompt = [{"role": "system", "content": prompt}]
        tool_prompt = [
            {"role": "system", "content": (
                f"Process the user's input using the tool '{tool.__name__}'. "
                f"User input: {messages[-1]['content']}, Topics: {topics}, Sentiment: {sentiment}"
            )}
        ]
        try:
            response = model.invoke(system_prompt + messages + tool_prompt)
            tool_result = tool(messages[-1]["content"])
            return {
                "messages": messages + [{"role": "assistant", "content": tool_result, "agent": name}],
                "selected_agent": name
            }
        except Exception as e:
            return {
                "messages": messages + [{"role": "assistant", "content": f"Sorry, something went wrong with {name}. Try again!"}],
                "selected_agent": name
            }

    graph = StateGraph(AgentState)
    graph.add_node(name, agent_node)
    graph.set_entry_point(name)
    graph.add_edge(name, END)
    return graph.compile(name=name)

# Initialize Chat Agent (Updated to include sub-agents)
def initialize_chat_agent():
    model = AzureChatOpenAI(client)

    # Create specialized agents
    personalized_plan_update_agent = create_custom_agent(
        model=model,
        tool=suggest_therapy_plan,
        name="personalized_plan_update_agent",
        prompt="You are an expert in creating and updating personalized therapy plans."
    )
    wellness_check_agent = create_custom_agent(
        model=model,
        tool=check_wellness,
        name="wellness_check_agent",
        prompt="You are a wellness monitoring expert."
    )
    post_rehab_follow_up_agent = create_custom_agent(
        model=model,
        tool=post_rehab_followup,
        name="post_rehab_follow_up_agent",
        prompt="You are a post-rehab follow-up expert."
    )

    # Dictionary of sub-agents for routing
    sub_agents = {
        "personalized_plan_update_agent": personalized_plan_update_agent,
        "wellness_check_agent": wellness_check_agent,
        "post_rehab_follow_up_agent": post_rehab_follow_up_agent
    }

    # Create chat agent with sub-agents
    chat_agent = create_chat_agent(model, sub_agents)
    return chat_agent

# Global chat agent instance
app.chat_agent = initialize_chat_agent()

# Updated Chat Endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_id = data.get("user_id")
    user_message = data.get("message", "").strip()

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    try:
        query = f"SELECT * FROM c WHERE c.user_id = '{user_id}'"
        user_docs = list(container.query_items(query, enable_cross_partition_query=True))
        if not user_docs:
            return jsonify({"error": "User not found"}), 404

        user_doc = user_docs[0]
        conversation_history = user_doc.get("conversation_history", [])
        user_name = user_doc.get("first_name", "friend")

        # Prepare initial messages
        if not user_message:
            initial_messages = conversation_history
        else:
            initial_messages = conversation_history + [{"role": "user", "content": user_message}]

        # Invoke the chat agent (which now handles routing)
        result = app.chat_agent.invoke({
            "messages": initial_messages,
            "user_name": user_name,
            "topics": [],
            "sentiment": "neutral",
            "selected_agent": "chat_agent"
        })

        # Extract response and agent
        last_message = result["messages"][-1]
        ai_response = last_message["content"]
        selected_agent = last_message.get("agent", "chat_agent")

        # Update conversation history
        if not user_message:
            conversation_history.append({"role": "assistant", "content": ai_response, "agent": selected_agent})
        else:
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": ai_response, "agent": selected_agent})
        user_doc["conversation_history"] = conversation_history
        container.upsert_item(user_doc)

        return jsonify({"response": ai_response, "agent": selected_agent})

    except exceptions.CosmosHttpResponseError as e:
        return jsonify({"error": f"CosmosDB Error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

@app.route("/")
def home():
    return "SoulSync API is running!"

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
