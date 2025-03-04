import telebot
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize the bot with your API key
bot = telebot.TeleBot("7799136952:AAFGeNLUwtWu7N7zpPRNUpL2JWa1NBYcXxY")  # Replace with your Telegram Bot API key

# Function to download and load the 7b-text-v0.2-q2_K model
def load_model():
    model_name = "7b-text-v0.2-q2_K"  # Replace this with the actual model path or name if different
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Load the model and tokenizer
model, tokenizer = load_model()

# Define a command handler
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! I'm your Mistral AI bot. Send me a message and I'll reply.")

# Define a message handler
@bot.message_handler(func=lambda message: True)
def generate_response(message):
    user_message = message.text

    # Tokenize the input text
    inputs = tokenizer.encode(user_message, return_tensors="pt")
    
    # Generate a response from the model
    with torch.no_grad():
        output = model.generate(inputs, max_length=50)  # Limit max length for better memory management
    
    # Decode the model's output
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Send the response back to the user
    bot.reply_to(message, response)

# Polling to keep the bot running
bot.polling()
