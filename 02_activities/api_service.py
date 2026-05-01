import os
from openai import OpenAI
from dotenv import load_dotenv

# Loading API key from secret.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(BASE_DIR, "../05_src/.secrets")
load_dotenv(dotenv_path, override=True)
API_KEY = os.getenv("API_GATEWAY_KEY")

# Creating an OpenAI client.
client = OpenAI(
    base_url = "https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
    api_key = os.getenv("API_GATEWAY_KEY"),
    default_headers = {"x-api-key": os.getenv("API_GATEWAY_KEY")}
)

# Creating a function to call an LLM API to process the user query, then 
# transform the output into a more friendly and concise response using 
# a custom function.
def call_api_service(user_query: str) -> str:
    """
    Service 1: Calls the API and transforms the response
    """

    try:
        # Step 1: Calling the API function.
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that rewrites answers in a friendly and concise way."
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ],
            temperature=0.7
        )

        raw_output = response.choices[0].message.content

        # Step 2: Transforming the response.
        transformed_response = transform_response(raw_output)

        return transformed_response

    except Exception as e:
        return f"Error calling API service: {str(e)}"

# Creating a function to rewrite the API response using an LLM to 
# improve tone, clarity, and conciseness, with a fallback to the 
# original text if it fails.
def transform_response(text: str) -> str:
    """
    Transform the API response (IMPORTANT for assignment requirement)
    Example transformations:
    - Make it more conversational
    - Shorten it
    - Rephrase tone
    """

    try:
        transformation_prompt = f"""
        Rewrite the following response in a more natural, friendly, and concise tone.
        Do NOT copy it verbatim. Make it sound like a helpful chatbot.

        Original:
        {text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a rewriting assistant."},
                {"role": "user", "content": transformation_prompt}
            ],
            temperature=0.5
        )

        return response.choices[0].message.content

    except Exception as e:
        # Need to use a fallback if transformation fails.
        return text


# Creating a few test samples for testing.
if __name__ == "__main__":
    query = "Explain what machine learning is."
    result = call_api_service(query)
    print("\nAPI Service Output:\n")
    print(result)
