import os
from dotenv import load_dotenv
from openai import OpenAI
from api_service import call_api_service
from semantic_service import semantic_search

# Getting the secret key.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "../05_src/.secrets")
load_dotenv(dotenv_path, override=True)
API_KEY = os.getenv("API_GATEWAY_KEY")

# Creating an OpenAI client.
client = OpenAI(
    base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key=API_KEY,
    default_headers={"x-api-key": API_KEY}
)

# Creating a calculator function.
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"The result is {result}"
    except Exception:
        return "Invalid math expression."


# Creating a variable to contain the data.
tools = [
    {
        "type": "function",
        "function": {
            "name": "use_api_service",
            "description": "Use this for general questions or explanations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "use_semantic_search",
            "description": "Use this for knowledge-based or ML-related questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Use this for math calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]


# Creating a function to handle logics.
def handle_user_query(user_query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a smart assistant that selects the correct tool."},
                {"role": "user", "content": user_query}
            ],
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        # Tool is called to use.
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name

            import json
            arguments = json.loads(tool_call.function.arguments)

            if function_name == "use_api_service":
                return call_api_service(arguments["query"])

            elif function_name == "use_semantic_search":
                return semantic_search(arguments["query"])

            elif function_name == "calculate":
                return calculator(arguments["expression"])

        # Tool is not used.
        return message.content

    except Exception as e:
        return f"Error in tool service: {str(e)}"


# Creating a few test functions for testing.
if __name__ == "__main__":
    queries = [
        "Explain machine learning",
        "What is deep learning?",
        "What is 25 * 4 + 10?"
    ]

    for q in queries:
        print("\nUser:", q)
        print("Response:", handle_user_query(q))

