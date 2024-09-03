# WATI Assistant

WATI Assistant (Wise Ai for Textual Interaction) is an AI-powered assistant with a variety of features, including conversational abilities, arithmetic and equation solving, weather updates, YouTube music playback, and more. This assistant uses a combination of pre-trained models, semantic similarity, and fallback mechanisms to provide accurate and helpful responses.

## Features

- **Conversational AI:** Uses a pre-trained language model for generating responses.
- **Arithmetic and Equation Solving:** Solves arithmetic expressions and algebraic equations.
- **Weather Updates:** Fetches weather information for specified locations.
- **YouTube Music Playback:** Plays, pauses, and stops music from YouTube.
- **Sentiment Analysis:** Analyzes the sentiment of user queries and provides emotionally responsive answers.
- **Feedback Mechanism:** Allows users to provide feedback and update answers in the dataset.
- **Simulated Typing Effect:** Displays responses with a simulated typing effect for a more human-like interaction.
- **User History Navigation:** Navigate through user input history with the up arrow key.
- **Dynamic Chat Interface:** Clear chat history, exit the application, and send queries through a user-friendly interface.

### Prerequisites

Ensure you have the following software installed on your system:

- Python 3.7 or higher
- pip (Python package installer)

### Required Libraries

The following Python libraries are required for the WATI Assistant to function:

- `tkinter`
- `pandas`
- `spacy`
- `re`
- `os`
- `math`
- `time`
- `random`
- `sympy`
- `concurrent.futures`
- `csv`
- `pytube`
- `youtube_search`
- `yt_dlp`
- `ffmpeg`
- `pyaudio`
- `requests`
- `threading`
- `textblob`
- `datetime`

You can install these libraries using pip:

```bash
pip install pandas spacy sympy pytube youtube-search yt-dlp pyaudio requests textblob
python -m spacy download en_core_web_sm
```

### Running the Application

1. **Clone the Repository** (if applicable) or download the script files to your local machine.
2. **Navigate to the Directory** containing the script files.
3. **Run the Script**:

```bash
python3 Ai.py
```

## Usage

Once the application is running, you can interact with WATI Assistant through the provided user interface. Type your questions or commands into the input field and press `Enter` or click the `Send` button. 

### Available Commands:

- **Arithmetic and Equations:** Ask math-related questions or provide equations to solve.
  - Example: `Solve 2 + 2`, `Solve x + 2 = 5`
- **Weather Updates:** Ask for weather information.
  - Example: `What is the weather in New York?`
- **YouTube Music Playback:** Play, pause, resume, or stop music from YouTube.
  - Example: `Play music Despacito`, `Stop music`
- **Clear Chat:** Type `clear chat` to clear the chat history.
- - **Aditional command:** Type `--no dataset` at the end of your question to answer without the default dataset.
- **Exit Application:** Type `exit` to close the application.

### Feedback Mechanism

If the assistant provides an incorrect answer, you can correct it by providing feedback:

1. Type `correct <your query>` to initiate the correction process.
2. Provide the corrected answer when prompted.

### Navigation

Use the up arrow key to navigate through your input history.

## License

This project is licensed under the GNU GPL License.

## Acknowledgements

- [spaCy](https://spacy.io/)
- [YouTube Data API](https://developers.google.com/youtube/v3)
- [OpenWeatherMap](https://openweathermap.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
