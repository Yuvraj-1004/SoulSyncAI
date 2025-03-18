import os
from langchain_openai import AzureChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Initialize Azure OpenAI Model
model = AzureChatOpenAI(
    azure_endpoint=os.getenv("ENDPOINT"),
    api_key=os.getenv("API_KEY"),
    api_version="2023-05-15",
    deployment_name="gpt-4o-mini"
)

# Create specialized agents
def add(a: float, b: float) -> float:
    """Add two numbers."""
    result = a + b
    print(f"[MATH_AGENT] Adding {a} + {b} = {result}")
    return result

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    result = a * b
    print(f"[MATH_AGENT] Multiplying {a} * {b} = {result}")
    return result

def web_search(query: str) -> str:
    """Search the web for information."""
    print(f"[RESEARCH_AGENT] Searching web for: {query}")
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

# Create agents
math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world-class researcher with access to web search. Do not do any math."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile workflow
app = workflow.compile()

# Function to test agent control flow
def test_workflow():
    test_cases = [
        ("What's the combined headcount of the FAANG companies in 2024?", "research"),
        ("What is 25 + 30?", "math"),
        ("Multiply 7 by 8.", "math"),
        ("Search for the latest AI trends.", "research"),
    ]
    
    for query, expected_agent in test_cases:
        print("\n" + "="*50)
        print(f"[TEST] User Query: {query}")
        print("="*50)
        
        try:
            result = app.invoke({"messages": [{"role": "user", "content": query}]})
            print("[SYSTEM RESPONSE]:", result)
        except Exception as e:
            print("[ERROR]:", str(e))

# Run the test workflow
test_workflow()
