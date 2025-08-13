import os
import openai
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
AIPROXY_BASE_URL = os.getenv("AIPROXY_BASE_URL")

# Initialize OpenAI client
client = openai.OpenAI(api_key=AIPROXY_TOKEN, base_url=AIPROXY_BASE_URL)

def get_llm_response(question: str, file_content: str = None):
    """
    Sends the question (and optional file content) to the LLM and retrieves a response.
    """
    try:
        prompt = prompt = f"""Answer the following question concisely. 
        Only return the final numerical or textual answer without any explanation or additional words.

        Question: {question}
        """
        if file_content:
            prompt += f"\n\nContext from file:\n{file_content}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust based on the model you are using
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )

        # Extract the answer from the response
        answer = response.choices[0].message.content.strip()

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}
