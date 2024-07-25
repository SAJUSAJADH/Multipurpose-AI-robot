import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('API_KEY'))


generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="Your job is to support my AI system by processing the user input and finding the appropriate function to execute from the list of functions I give. Just return the selected function name/names only.",
)

history = []


def gen_ai(user_input):

    print('generating response...')
 
    chat_session = model.start_chat(
        history=history
    )

    response = chat_session.send_message(f"user input: {user_input} and the function and thier duty list is - {[]}")

    model_response = response.text

    history.append({"role": "user", "parts": [user_input]})
    history.append({"role": "model", "parts": [model_response]})

    return model_response

    



