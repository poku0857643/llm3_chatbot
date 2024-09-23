# from django.shortcuts import render
# from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
# import torch
# from django.http import JsonResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from django.conf import settings
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import os
# import time
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from django.conf import settings
# import threading
# from .load_and_run import *
# from .model_choice import generate_response

#
# # Initialize global variable for the models and tokenizers
# tokenizers = {}
# models = {}
# embeddings = None
# document_embeddings = None
#
# # Detect if MPS is available, otherwise use CPU
# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
# print(f"Model loaded onto device: {device}.")
#
#
#
#
# load_dotenv()
#
# # Load Hugging Face token from environment variable
# hf_token = os.environ.get('HUGGINGFACE_TOKEN')
# #hf_token = settings.HUGGINGFACE_TOKEN
# if not hf_token:
#     raise EnvironmentError("Hugging Face token not found. Please set 'HUGGINGFACE_TOKEN'.")
#
# # # example corpus of documents
# # documents = settings.PAPERS_DIR
#
# # List of available models
# available_models = ['llama3', 'ft_rag']
#
#
# def your_view(request):
#     if request.method == 'POST':
#         prompt = request.POST.get('prompt')
#         model_choice = request.POST.get('model_choice')
#
#         # Generate the response
#         response_text = generate_response(prompt, model_choice)
#
#         # Pass the response_text to your HTML template
#         return render(request, 'your_template.html', {'response_text': response_text})
#     else:
#         return render(request, 'your_template.html')
#
# def chat_interface(request):
#     return render(request, 'chatbot.html')
#
#
# @csrf_exempt
# def chatbot_response(request):
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#             user_message = data.get('message', '').strip()
#             model_choice = data.get('model', 'llama3').strip() # Default to 'llama3'
#
#             # Check if the user_message is provided
#             if not user_message:
#                 return JsonResponse({'error': 'No message provided'}, status=400)
#
#             if model_choice not in models or model_choice not in tokenizers:
#                 return JsonResponse({'error': f"Model 'model_choice' not available"}, status=400)
#
#             model = models[model_choice]
#             tokenizer = tokenizers[model_choice]
#
#             # Ensure tokenizer has a unique pad_token
#             if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
#                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#                 model.resize_token_embeddings(len(tokenizer))
#
#             # Print the user_message to the console.log for debugging
#             print(f"Received message: {user_message}")
#             print(f"Using model: {model_choice}")
#
#             if model_choice == 'ft_rag':
#                 # Ensure embeddings_model is loaded
#                 if embeddings is None or document_embeddings is None:
#                     return JsonResponse({'error': 'Embeddings model not loaded or no document embeddings'}, status=500)
#
#                 # Retrieve relevant documents
#                 retrieved_docs = retrieve_documents(user_message)
#                 # Combine retrieved documents into a sigle context
#                 context = "\n".join(retrieved_docs)
#                 # Create the input text with context
#                 input_text = f"Question: {user_message}\nContext: {context}\nAnswer:"
#             else:
#                 # For 'llama3' or other models
#                 input_text = "Question: " + user_message
#
#             # Tokenize and generate a response
#             inputs = tokenizer(
#                 input_text,
#                 return_tensors='pt',
#                 padding=True,
#                 truncation=True,
#                 max_length=512)
#
#             inputs = inputs.to(model.device)
#             input_ids = inputs.input_ids
#             attention_mask = inputs.attention_mask
#
#             # Create a textiteratorstreamer for streaming generation
#             streamer = TextIteratorStreamer(
#                 tokenizer,
#                 skip_prompt=True,
#                 skip_special_tokens=True
#             )
#
#             # Define generation parameters
#             generation_kwargs = dict(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 max_new_tokens=150,
#                 streamer=streamer,
#                 do_sample=True,  # Enable sampling if you want more varied outputs
#                 temperature=0.7,  # Adjust temperature for creativity
#                 top_k=50,         # Consider top_k tokens at each step
#                 top_p=0.95,       # Nucleus sampling
#                 no_repeat_ngram_size=2,
#                 early_stopping=True
#             )
#
#
#             # Start the generation in a separate thread
#             generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
#             generation_thread.start()
#
#             generated_words = []
#             current_word = ''
#
#             # Iterate over the streamer to get tokens as they are generated
#             for new_token in streamer:
#                 # Accumulate characters to form words
#                 current_word += new_token
#
#                 # Check if the current token ends with a space or punctuation, indicating end of a word
#                 if new_token.strip() == '' or new_token in ['.', ',', '!', '?', ';', ':', '\n']:
#                     if current_word.strip() != '':
#                         # Append the word to the list and reset current_word
#                         generated_words.append(current_word)
#                         current_word = ''
#                 else:
#                     # Continue accumulating characters
#                     pass
#
#             # Ensure any remaining word is added
#             if current_word.strip() != '':
#                 generated_words.append(current_word)
#
#             # Wait for the generation thread to finish
#             generation_thread.join()
#
#             # Combine the generated words into a single string
#             bot_message = ''.join(generated_words)
#
#             # output = model.generate(
#             #     inputs['input_ids'],
#             #     max_length=150,
#             #     num_return_sequences=1,
#             #     no_repeat_ngram_size=2,
#             #     early_stopping=True)
#             #
#             # bot_message = tokenizer.decode(output[0], skip_special_tokens=True)
#
#             # Optionally, post-process the bot_message to remove the prompt text
#             if model_choice == 'ft_rag':
#                 # Remove the input_text from the generated text
#                 bot_message = bot_message.replace(input_text, '').strip()
#             else:
#                 bot_message = bot_message.replace(f"Question: {user_message}", '').strip()
#
#             return JsonResponse({'bot_message': bot_message}, status=200)
#
#         except json.JSONDecodeError as e:
#             return JsonResponse({'error': 'Invalid JSON provided'}, status=400)
#         except Exception as e:
#             print(f"Error during model processing: {e}")
#             return JsonResponse({'error': f"Model processing failed"}, status=500)
#
#
#
#     # Handle non-POST requests
#     return JsonResponse({'error': 'Invalid request method'}, status=400)
#
# #############################################
# # Call the function to load the models and tokenizers at startup
# # load_models_and_tokenizers()

###########
#ft_model_only

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from llm3_chatbot import settings
import os
from dotenv import load_dotenv
import threading
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
import time
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import StreamingHttpResponse, HttpResponseServerError
from django.utils.safestring import mark_safe
from transformers import TextIteratorStreamer


# Set up device
device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

# # List of available models
available_models = ['llama3', 'ft_rag']

model_path = '/Users/eshan/PycharmProjects/llm3_chatbot/model/llm_model'
tokenizer_path = '/Users/eshan/PycharmProjects/llm3_chatbot/model/llm_tokenizer'


model = LlamaForCausalLM.from_pretrained(model_path, use_safetensors=True)
model.to(device)
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)


# Ensure tokenizer has a unique pad_token
if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

text_generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    max_new_tokens = 128,
    device=device
)

def chat_interface(request):
    return render(request, 'chatbot.html')


def get_responsew2w(prompt):
    # Tokenize the input prompt
    inputs = (tokenizer(prompt, return_tensors="pt",padding=True, truncation=True, max_length=512))
    inputs = inputs.to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Create a TextIteratorStreamer for streaming generation
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    # Define generation parameters
    generation_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        streamer=streamer,
        do_sample=True,  # Enable sampling if you want more varied outputs
        temperature=0.7,  # Adjust temperature for creativity
        top_k=50,  # Consider top_k tokens at each step
        top_p=0.95  # Nucleus sampling
    )

    # Start the generation in a separate thread
    generation_thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    generation_thread.start()


    current_word = ''

    # Iterate over the streamer to get tokens as they are generated
    for new_token in streamer:
        # Accumulate characters to form words
        current_word += new_token

        # Check if the current token ends with a space or punctuation, indicating end of a word
        if new_token.strip() == '' or new_token in ['.', ',', '!', '?', ';', ':', '\n']:
            if current_word.strip() != '':
                # Output the word and reset current_word
                print(current_word, end='!', flush=True)
                word = current_word.strip()
                yield word
                current_word = ''
        elif new_token.strip() == '':
            continue
        else:
            # For tokens that don't end a word, continue accumulating
            pass

    # Ensure any remaining word is added
    if current_word.strip() != '':
        print(current_word, end='', flush=True)
        #word = current_word.strip
        yield current_word
#        generated_words.append(current_word)
#     return current_word
    # Wait for the generation thread to finish
        generation_thread.join()





@csrf_exempt
def chatbot_response(request):
    try:
    # if request.method == 'POST':
        if request.method == 'POST':
            # try:
                data = json.loads(request.body)
                user_message = data.get('message', '').strip()
                # model_choice = data.get('llama3', 'ft_rag').strip() # Default to 'llama3'

                # Check if the user_message is provided
                if not user_message:
                    return HttpResponseServerError({'error': 'No message provided'}, status=400)

                # Generate the response using get_responsew2w
                print("user message:", user_message)
                bot_message_iterator = get_responsew2w(user_message)
                print("bot_message",bot_message_iterator)

                # Define a generator function for the StreamingHttpResponse
                def event_stream(bot_message_iterator):

                    # word_buffer = []
                    for word in bot_message_iterator:
                        # word_buffer.append(word)
                        # if len(word_buffer) == 1:
                        #     data = ' '.join(word_buffer)
                            #print(f"Sending word to client: {word}", flush=True)
                        # SSE requires data to be sent in a specific format
                            yield f'data: {word}\n\n'
                            #word_buffer = []
                    # if word_buffer:
                    #     data = ' '.join(word_buffer)
                    #     yield f'data: {data}\n\n'



                # Return a StreamingHttpResponse with content type 'text/event-stream'
                response = StreamingHttpResponse(event_stream(bot_message_iterator), content_type='text/event-stream')
                print("response", response)
                print("data", data)
                response['Cache-Control'] = 'no-cache'
                response['model-status'] = 'loaded'  # Set the 'model-status' header
                #response['Access-Control-Allow-Origin'] = '*'
                return response
        else:
            return JsonResponse({'error': 'Unsupported method'}, status=405)


    except Exception as e:
        print(f"Error in chatbot_response view: {e}")
        return HttpResponseServerError("Internal Server Error")







