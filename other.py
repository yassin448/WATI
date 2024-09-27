
##########################################################################################################################
################################################IMPORT SECTIONS###########################################################
##########################################################################################################################

import tkinter as tk
from tkinter import scrolledtext
from PIL import Image , ImageTk
import pandas as pd
import spacy
import re
import os
import math
import time
import random
from sympy import symbols, Eq, solve
import concurrent.futures
import Aidata as fallbackai
import csv
from pytube import YouTube
from youtube_search import YoutubeSearch
import yt_dlp
import ffmpeg
import pyaudio
import requests
import threading
from textblob import TextBlob
from datetime import datetime
import WATI_finetuned as WATI_core
from datasets import load_dataset
from transformers import T5Tokenizer
from time import sleep
import socket


##########################################################################################################################
################################################GLOBAL VARIABLES##########################################################
##########################################################################################################################


# Initialize tokenizer
tokenizer = T5Tokenizer.from_pretrained("/home/yassin/Documents/Aiwithypython/WATI_core_dependencies")

# Load daily_dialog dataset
dataset = load_dataset("daily_dialog")
train_data = dataset["train"]
validation_data = dataset["validation"]

st = None
message = None


# Global variable to hold instance of YouTubePlayer
youtubeplayer = None

#global history
history = None

#global question
question = None

#is answering
isanswering =False

#server mode globals
client_socket = None

# Load spaCy model with word vectors (for semantic similarity)
nlp = spacy.load("en_core_web_sm")


# OpenWeatherMap API credentials
WEATHER_API_KEY = 'c27ea1d81b6a244f89ce37c2c535330c'

# Model information
MODEL_NAME = "WATI"
MODEL_VERSION = "2.5"


##########################################################################################################################
#################################################LOCAL CLASSES###########################################################
##########################################################################################################################

# YouTubePlayer class to handle music playback
class YouTubePlayer:
    def __init__(self):
        self.process = None
        self.audio_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.current_url = None

    def get_youtube_audio_url(self, query):
        yt_search = YoutubeSearch(query, max_results=1).to_dict()
        if not yt_search:
            return None

        yt_url = f"https://www.youtube.com/watch?v={yt_search[0]['id']}"

        ydl_opts = {
            'format': 'bestaudio/best',
            'quiet': True,
            'noplaylist': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(yt_url, download=False)
            audio_url = info_dict['url']

        return audio_url
    
    
    def play_stream(self, audio_url):
        self.process = (
            ffmpeg
            .input(audio_url)
            .output('pipe:', format='wav')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        p = pyaudio.PyAudio()
        self.stream_out = p.open(format=pyaudio.paInt16,
                                 channels=2,
                                 rate=44100,
                                 output=True)

        while not self.stop_event.is_set():
            while self.pause_event.is_set() : pass
            data = self.process.stdout.read(4096)
            #print(self.stop_event)
            if not data:
                break
            self.stream_out.write(data)

        self.stream_out.stop_stream()
        self.stream_out.close()
        p.terminate()
        self.process.stdout.close()

    def play(self, query):
        audio_url = self.get_youtube_audio_url(query)
        if audio_url is None:
            print("No results found for the query.")
            return 
        self.pause_event.clear()
        self.stop_event.clear()
        self.audio_thread = threading.Thread(target=self.play_stream, args=(audio_url,))
        
        self.audio_thread.start()

      
        return f"Playing '{query}' on YouTube Music."

    def stop(self):
        if self.process:
            
            self.process = None
            self.stop_event.set()
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join()
            self.audio_thread = None

    def pause(self):
        if self.process and not self.pause_event.is_set():
            self.pause_event.set()
            return "Music playback paused."
        return "No music is currently playing."

    def resume(self):
        if self.process and self.pause_event.is_set():
            self.pause_event.clear()
            
            return "Music playback resumed."
        return "No music is currently paused."



##########################################################################################################################
################################################LOCAL FUNCTIONS###########################################################
##########################################################################################################################

def preprocess_function(examples):
    inputs = examples['dialog'][:-1]  # All but the last line
    targets = examples['dialog'][1:]  # All but the first line
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




# Ensure feedback.csv file with required headers
def ensure_feedback_csv():
    feedback_file = 'feedback.csv'
    if not os.path.exists(feedback_file):
        df = pd.DataFrame(columns=['Prompt', 'Corrected Answer'])
        df.to_csv(feedback_file, index=False)
        print(f"Created {feedback_file} with required headers.")

# Function to preprocess text using spaCy NLP
def preprocess_text(text):
    doc = nlp(text.lower())
    processed_text = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    return processed_text

# Function to load dataset with data review mechanism
def load_dataset(file_path):
    
    try:
        data = pd.read_csv(file_path)
        data.drop_duplicates(inplace=True)
        if data.isnull().any().any():
            raise ValueError("Dataset contains missing values.")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return pd.DataFrame(columns=['ques', 'ans'])

# Function to update feedback CSV
def update_feedback_csv(prompt, corrected_answer):
    feedback_file = 'feedback.csv'
    fieldnames = ['Prompt', 'Corrected Answer']
    try:
        with open(feedback_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'Prompt': prompt, 'Corrected Answer': corrected_answer})
    except IOError:
        print(f"Error writing to {feedback_file}")

# Function to update dataset entry or add new entry
def update_entry(dataset_path, question, answer):
    try:
        data = pd.read_csv(dataset_path)
        processed_question = preprocess_text(question)
        existing_entry = data['ques'].apply(preprocess_text) == processed_question
        if existing_entry.any():
            data.loc[existing_entry, 'ans'] = answer
            print(f"Updated existing entry for question: {question}")
        else:
            new_entry = pd.DataFrame({'ques': [question], 'ans': [answer]})
            data = pd.concat([data, new_entry], ignore_index=True)
            print(f"Added new entry for question: {question}")
        data.to_csv(dataset_path, index=False)
        print("Dataset updated successfully.")
    except pd.errors.EmptyDataError:
        new_data = pd.DataFrame({'ques': [question], 'ans': [answer]})
        new_data.to_csv(dataset_path, index=False)
        print("Dataset was empty. Created new dataset with the provided entry.")
    except FileNotFoundError:
        new_data = pd.DataFrame({'ques': [question], 'ans': [answer]})
        new_data.to_csv(dataset_path, index=False)
        print("Dataset file not found. Created new dataset with the provided entry.")
    except Exception as e:
        print(f"Error updating dataset entry: {e}")

# Function to solve arithmetic/math questions using NLP
def solve_math(question):
    question=str(question).lower().replace("solve","")
    math_pattern = r'([\d+\-*/\s\^()]+|sqrt\(\d+\)|log\(\d+\)|sin\(\d+\)|cos\(\d+\)|tan\(\d+\))'
    match = re.search(math_pattern, question)
    if match:
        try:
            result = eval(match.group(1), {"__builtins__": None}, {"sqrt": math.sqrt, "log": math.log, "sin": math.sin, "cos": math.cos, "tan": math.tan})
            return f"The result of '{match.group(1)}' is {result}."
        except Exception as e:
            print(f"Error solving math expression: {e}")
            return None
    return None

# Function to solve algebraic equations using NLP
def solve_equation(question):
    equation_pattern = r'([a-zA-Z]+\s*=\s*[\d+\-*/\s\^()]+|[\d+\-*/\s\^()]+\s*=\s*[a-zA-Z]+)'
    matches = re.findall(equation_pattern, question)
    if matches:
        try:
            eq = matches[0].replace('=', '==')
            lhs, rhs = eq.split('==')
            var = re.findall(r'[a-zA-Z]', lhs + rhs)[0]
            var = symbols(var)
            equation = Eq(eval(lhs), eval(rhs))
            solution = solve(equation, var)
            return f"The solution to the equation is {solution}."
        except Exception as e:
            print(f"Error solving equation: {e}")
            return None
    return None

# Function to get answer from dataset with best matching question
def get_answer(question, data):
    processed_question = preprocess_text(question)
    if not data.empty:
        data['similarity'] = data['ques'].apply(lambda x: nlp(processed_question).similarity(nlp(preprocess_text(x))))
        best_match_index = data['similarity'].idxmax()
        best_similarity = data.loc[best_match_index, 'similarity']
        if best_similarity > 0.8:
            return data.loc[best_match_index, 'ans']
    return None





def handle_question_mode_switch(question, entry):
    global st

    if "switch to server mode" in question.lower():
        

        response = f"{MODEL_NAME} v{MODEL_VERSION}: server mode enabled"
        simulate_typing(response)
        start_ai_server_set_event()
        return

    elif "switch to normal mode" in question.lower():
        if st and st.running():  # Check if the server is still running
            st.result(timeout=None)  # Wait for server shutdown

        response = f"{MODEL_NAME} v{MODEL_VERSION}: normal mode enabled"
        simulate_typing(response)
        start_ai_server_clear_event()
        return

def server_receive_messages():
    global client_socket
    try:
        # Receive message from client
        message = client_socket.recv(4096).decode()
        #print (f'received messahe from server side {message}')
        if not message:
            #print("Client disconnected")
            pass
        else :
            # Process the received message
            #print(f"\nReceived from client: {message}")
            handle_user_input(message)
            
            
        #print(f"\nClient: {message}")
        
        
    except Exception as e:
        print(f"Error receiving message: {e}")
        

def server_send_messages(user_input):
    global client_socket
    try:
        client_socket.send(user_input.encode())
    except Exception as e:
        print(f"Error sending message: {e}")
        

def start_ai_server_set_event ():
    global server_event
    server_event.set()
    
def start_ai_server_clear_event ():
    global server_event
    server_event.clear()
    
def start_ai_server():
    
    global client_socket , server_event
    server_event.wait()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # To reuse the address
    host = '0.0.0.0'  # Bind to all available interfaces
    port = 8080

    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"Server is listening on port {port}...")
    except Exception as e:
        print(f"Bind failed: {e}")
        return

    try:
        client_socket, addr = server_socket.accept()
        print(f"Client connected from {addr}")
    except Exception as e:
        print(f"Failed to accept connection: {e}")
        return

    while server_event.is_set() : 
        server_receive_messages()
        
    # Close sockets
    client_socket.close()
    server_socket.close()


def tell_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time
def correctQuestion(question):
    if "correct" in question.lower():
        str(question).replace("correct ","")
        blob = TextBlob(question)
        response = blob.correct()
        
        return response

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_emotional_response(sentiment,answer):
    if sentiment > 0.5:
        return f"{answer} 😊"
    elif sentiment > 0:
        return f"{answer}🙂"
    elif sentiment == 0:
        return f"{answer} 😐"
    elif sentiment > -0.5:
        return f"{answer} 😟"
    else:
         return f"{answer} 😢"


def write_to_chat_area(response):
    
    # Simulate typing effect by adding random sleep times between characters for UI
    for char in response:
        chat_area.insert(tk.END, char)
        chat_area.update_idletasks()
        time.sleep(random.uniform(0.02, 0.08))
    chat_area.insert(tk.END, "\n")
    
# create a new thread
def simulate_typing(answer):
    #write response to server if active
    global server_event
    if server_event.is_set() : 
        server_send_messages(answer)
    else :
        pass
    t = threading.Thread(target=write_to_chat_area , args=(answer,))
    t.start()
    
    
    

# Function to clear the chat area and input
def clear_chat():
    chat_area.delete('1.0', tk.END)
    entry.delete(0, tk.END)
    chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Welcome to WATI Assistant! How can I help you today?\n")

def _on_mousewheel(self, event):
    self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
# Function to exit the application
def exit_app():
     
     root.quit()  # Close the Tkinter window
     root.destroy()  # Destroy the root window
     os._exit(0)  # Exit the Python interpreter
     

# Function to get weather information
def get_weather(question):
    try:
        # Extract city name from the question
        location = extract_location(question)
        if location:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url)
            weather_data = response.json()
            if weather_data['cod'] == 200:
                description = weather_data['weather'][0]['description']
                temperature = weather_data['main']['temp']
                return f"The weather in {location} is {description} with a temperature of {temperature}°C."
            else:
                return f"Sorry, I couldn't find the weather information for {location}."
        else:
            return "Please specify a location to get the weather information."
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return "Sorry, there was an error fetching the weather information."

# Function to extract location from the question
def extract_location(question):
    try:
        # Remove punctuation and lowercase the question for uniformity
        clean_question = re.sub(r'[^\w\s]', '', question.lower())
        # Split the question into words and look for common location indicator
        words = clean_question.split()
        if 'weather' in words and 'in' in words:
            # Find the index of 'in' and get the word after it as location
            index = words.index('in')
            return words[index + 1]
        else:
            return None
    except Exception as e:
        print(f"Error extracting location: {e}")
        return None

# Function to handle user input
def handle_user_input(question):
    entry.delete(0, tk.END)
    chat_area.insert(tk.END, f"You: {question}\n")
    chat_area.update_idletasks()
         
    # Check for stop playback command
    if question.lower() in ["stop music", "stop song", "stop playback"]:
        if youtubeplayer:
            youtubeplayer.stop()
            response = "Music playback stopped."
        else:
            response = "No music is currently playing."
        simulate_typing(response)
        return
    elif "tell time" in  question.lower():
        response = datetime.now().strftime("%A, %d. %B %Y %I:%M:%S %p")
        simulate_typing(response)
        return
    # Check for translation command
    elif "translate to" in question.lower():
        # Extract the target language and text to translate
        parts = question.split(":")
        target_language = parts[0].strip().lower().replace("translate to", "").strip()
        question = parts[1].strip() if len(parts) > 1 else ""
        prequestion = f"translate to {target_language}:"
        response = WATI_core.get_WATI_response(prequestion,question)
        
        simulate_typing(response)
        return
    # Check for pause playback command
    elif question.lower() in ["pause music", "pause song", "pause playback"]:
        if youtubeplayer:
            response = youtubeplayer.pause()
        else:
            response = "No music is currently playing."
        simulate_typing(response)
        return
    elif question.lower() in "switch to server mode":
       handle_question_mode_switch(question,entry)
       
    elif "--use-core" in question.lower():
        question = question.replace("--use-core", "")
        
        # Check if the colon is present before splitting
        if ':' in question:
            try:
                # Split the string into inprequestion and inquestion
                inprequestion, inquestion = question.split(':', 1)
                
                # Strip any leading or trailing whitespace
                inprequestion = inprequestion.strip()
                inquestion = inquestion.strip()
                
                chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}:Please note that if you use this mode, you need to add the specific task you want the AI to perform before asking the question to get accurate responses. For example: translate to german:hello\n")
                sleep(1)
                # Process response
                response = WATI_core.get_WATI_response(inprequestion, inquestion)
                
                # Format the answer
                sentiment = analyze_sentiment(response)
                answer = f"{MODEL_NAME} {MODEL_VERSION}: {get_emotional_response(sentiment, response)}"
                
                simulate_typing(answer + " Note that this WATI_core version is in beta and may fail to generate responses or provide inaccurate responses.")
                
            except ValueError:
                simulate_typing("Error: The question format is incorrect. Ensure you have a colon ':' separating the task and the question.")
                
        else:
            simulate_typing("Error: No colon ':' found in the input. Please format your input as 'task: question'.")
  
        
    elif "--use-groq" in question.lower():
        question = question.replace("--use-groq", "")
        # Fallback to Groq model for answering
        response = fallbackai.get_Groq_response(question)
        sentiment = analyze_sentiment(question)
        answer = get_emotional_response(sentiment,response)
        simulate_typing(answer)
        return
        
    # Check for exit command
    elif question.lower() == "exit":
     exit_app()
     return

    # Check for clear chat command
    elif question.lower() == "clear chat":
        clear_chat()
        return

    elif "correct" in question.lower():
        response = correctQuestion(question)
        answer = response
    
    # Check for weather query
    elif "weather" in question.lower():
        response = get_weather(question)
        simulate_typing(response)
        return

    # Check if input is a song request
    elif "play music" in question.lower():
        if youtubeplayer:
            song_query = question.replace("play", "").strip()
            response = youtubeplayer.play(song_query)
        else:
            response = "YouTube player is not initialized."
        simulate_typing(response)
        return
    else:
        # Uses Multithreading for quick answering
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            future_math = executor.submit(solve_math, question)
            future_equation = executor.submit(solve_equation, question)
            future_answer = executor.submit(get_answer, question, data)
            
            math_answer = future_math.result()
            equation_answer = future_equation.result()
            dataset_answer = future_answer.result()
            
            
            if math_answer:
                answer = math_answer
                simulate_typing(answer)
            elif equation_answer:
                answer = equation_answer
                simulate_typing(answer)
            elif dataset_answer:
                sentiment = analyze_sentiment(question)
                answer = MODEL_NAME+" v"+MODEL_VERSION+" Dataset: " + get_emotional_response(sentiment,dataset_answer)
                simulate_typing(answer)
            else:
                # Fallback to Groq model for answering
                response = fallbackai.get_Groq_response(question)
                sentiment = analyze_sentiment(question)
                answer = MODEL_NAME+" "+MODEL_VERSION+": "+get_emotional_response(sentiment,response)
                simulate_typing(answer)
                
                 

# Function to handle user feedback
def handle_feedback(event=None):
    feedback_input = entry.get().strip().lower()
    entry.delete(0, tk.END)

    if feedback_input in ["yes", "no"]:
        if feedback_input == "no":
            # Request for correct answer from user
            chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Please provide the correct answer.\n")
            entry.bind("<Return>", handle_corrected_answer)
        elif feedback_input =='yes':
            chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Thank you for your feedback!\n")
            entry.bind("<Return>", handle_user_input)
    else:
        chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Please answer with 'yes' or 'no'.\n")

# Function to handle corrected answer from user
def handle_corrected_answer(event=None):
    corrected_answer = entry.get().strip()
    entry.delete(0, tk.END)

    # Get the last user query from the chat area
    chat_text = chat_area.get('1.0', tk.END).strip().split('\n')
    last_user_query = next((line for line in reversed(chat_text) if line.startswith("You: ")), None)
    if last_user_query:
        last_user_query = last_user_query.replace("You: ", "")
        update_entry('token.csv', last_user_query, corrected_answer)
        update_feedback_csv(last_user_query, corrected_answer)
        chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Thank you! The answer has been updated.\n")
    else:
        chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Sorry, I couldn't find the previous question.\n")
    
    entry.bind("<Return>", handle_user_input)

# Function for when "Enter" key is pressed
def on_enter(event):
    global question, history_index
    question = entry.get().strip()
    if question:
        handle_user_input(question)
        history.append(question)  # Append the current question to history
        history_index = len(history)  # Reset history index
        entry.delete(0, tk.END)
        


# Function to handle up arrow key press
def on_up_arrow(event):
    global history_index
    if history:
        history_index = (history_index - 1) % len(history)
        entry.delete(0, tk.END)
        entry.insert(0, history[history_index])





##########################################################################################################################
#################################################MAIN FUNCTION############################################################
##########################################################################################################################


def main():
    global chat_area , history , history_index , root , entry , data, profile_pic_frame, main_frame
    # Load WATI_finetuned model
    WATI_core.loadmodel()

    # Init YouTubePlayer instance
    youtubeplayer = YouTubePlayer()

    # Ensure CSV and dataset are loaded correctly
    dataset_path = 'token.csv'
    data = load_dataset(dataset_path)
    ensure_feedback_csv()

    # Create main window
    root = tk.Tk()
    root.title("WATI Assistant")
    root.configure(bg='#343541')

    # Full UI frame (to hold the full UI)
    main_frame = tk.Frame(root, bg='#343541')
    main_frame.pack(fill=tk.BOTH, expand=True)

    # History to store user inputs
    history = []
    history_index = -1  # Index to navigate through history

    # Create chat display area
    chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, bg='#40414f', fg='#d1d5db', font=('Helvetica', 12), padx=10, pady=10, insertbackground='white', highlightthickness=0, borderwidth=0)
    chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    chat_area.insert(tk.END, f"{MODEL_NAME} {MODEL_VERSION}: Welcome to WATI Assistant! How can I help you today?\n")

    entry_frame = tk.Frame(main_frame, bg='#343541')
    entry_frame.pack(padx=10, pady=10, fill=tk.X)
    entry = tk.Entry(entry_frame, bg='#40414f', fg='#d1d5db', font=('Helvetica', 12), insertbackground='white', highlightthickness=0, borderwidth=0)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    entry.bind("<Return>", on_enter)
    entry.bind('<Up>', on_up_arrow)  # Bind the up arrow key
    chat_area.bind_all("<MouseWheel>", _on_mousewheel)

    # Create send button
    send_button = tk.Button(entry_frame, text="Send", command=lambda: on_enter(None), bg='#4f525d', fg='white', font=('Helvetica', 12), highlightthickness=0, borderwidth=0)
    send_button.pack(side=tk.RIGHT, padx=5)

    # Create clear button
    clear_button = tk.Button(main_frame, text="Clear", command=clear_chat, bg='#4f525d', fg='white', font=('Helvetica', 12), width=10, highlightthickness=0, borderwidth=0)
    clear_button.pack(padx=10, pady=10)

    # Create exit button
    exit_button = tk.Button(main_frame, text="Exit", command=exit_app, bg='#4f525d', fg='white', font=('Helvetica', 12), width=10, highlightthickness=0, borderwidth=0)
    exit_button.pack(padx=10, pady=10)

    # Load profile picture image
    profile_pic_frame = tk.Frame(root, width=100, height=100)  # Mini image size
    load_profile_picture(root, profile_pic_frame)


    
    # Start server thread in background
    server_thread = threading.Thread(target=start_ai_server)

    # Start the server in a separate thread
    server_thread.start()

    # Set a timer to minimize the UI
    minimize_after_timeout(5000)
    
    
    
    # Start the UI loop
    root.mainloop()

    # Wait for all threads to join
    server_thread.join()

def load_profile_picture(root, profile_pic_frame):
    """Load and display the profile picture in minimized mode."""
    # Load the image 
    image = Image.open("Aiimg.png")  
    image = image.resize((100, 100), Image.Resampling.LANCZOS)  # Resize to mini size
  
    profile_image = ImageTk.PhotoImage(image)

    # Create a label to hold the image
    profile_pic_label = tk.Label(profile_pic_frame, image=profile_image)
    profile_pic_label.image = profile_image 
    profile_pic_label.pack()

def minimize_after_timeout(timeout):
    """Set a timer to minimize the window after the timeout."""
    root.after(timeout, minimize_ui)

def minimize_ui():
    """Minimize the UI to a profile picture."""
    main_frame.pack_forget()  # Hide the full UI
    profile_pic_frame.pack(fill=tk.BOTH, expand=True)  # Show the profile picture
    root.geometry("100x100")  # Resize the window to match the image

    # Bind click event to expand the UI when the profile picture is clicked
    profile_pic_frame.bind("<Button-1>", expand_ui)

def expand_ui(event=None):
    """Expand the UI back to the full view."""
    profile_pic_frame.pack_forget()  # Hide the profile picture
    main_frame.pack(fill=tk.BOTH, expand=True)  # Show the full UI
    root.geometry("400x300")  # Restore the full window size
    minimize_after_timeout(5000)  # Reset the timer for the next minimization

if __name__ == "__main__":
    main()
