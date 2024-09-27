import os
from groq import Groq

# Initialize the Groq client
client = Groq(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # open groq cloud website and obtain your api key there
)

# Initialization message
initial_message = {
    "role": "system", 
    "content": "You are an AI assistant created by Yassin.Your Name is WATI(Wise Ai For Textual Intelligence).Your purpose is to assist users with their queries."
}

# Initialize conversational memory with the initial message
conversation_history = [initial_message]

def get_token_length(text):
    return len(text.split())

def truncate_conversation_history():
    # Ensure the total token length is within the limit
    total_length = sum(get_token_length(message["content"]) for message in conversation_history)
    
    while total_length > 16384:
        removed_message = conversation_history.pop(1)  # Remove the oldest user/assistant message, keep initial message
        total_length -= get_token_length(removed_message["content"])

def get_Groq_response(user_input):
    try:
        # Add the user message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})
        
        # Truncate conversation history if necessary
        truncate_conversation_history()
        
        
        # Get a response from the model
        chat_completion = client.chat.completions.create(
            messages=conversation_history,
            model="llama3-8b-8192",
        )

        # Extract the model's response
        response = chat_completion.choices[0].message.content

        # Add the model's response to the conversation history
        conversation_history.append({"role": "assistant", "content": response})
        
        #print("Groq Response:", response)  # Debugging statement
        
        return response

    except Exception as e:
        print(f"Error getting WATI response: {e}")
        return "I'm sorry, there was an error processing your request."
