import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def loadmodel():
 global model
 global tokenizer
 global device
 # Define the checkpoint directory
 checkpoint_dir = "./WATI_core_dependencies"

 # Load the tokenizer and model from the checkpoint
 tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
 model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)
 # Ensure the model is on the correct device
 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 model.to(device)





def get_WATI_response(inprequesion,inquestion):
    
    prequestion = inprequesion
    question = inquestion
    finalquestion = prequestion+question
    # Tokenize the input question
    inputs = tokenizer.encode(finalquestion, return_tensors="pt", truncation=True, padding="max_length", max_length=100)

    # Move inputs to the correct device
    inputs = inputs.to(device)

    # Generate the answer 
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=200,  # Increase max length to allow longer responses
            num_beams=7,  # Higher number of beams for better search
            early_stopping=True,  # Stop early if all beams end
            no_repeat_ngram_size=3,  # Prevent repetition of n-grams
            repetition_penalty=2.0,  # Penalize repeated phrases
            length_penalty=1.2,  # Encourage longer responses
        )

    # Decode the output to text
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer


