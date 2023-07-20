from transformers import pipeline
import gradio as gr

# Load the model
chat_pipeline = pipeline("text2text-generation", model="TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ")

def chat_with_model(input_text):
    # Generate a response
    result = chat_pipeline(input_text)
    # Get just the text from the returned dictionary
    response_text = result[0]['generated_text']
    return response_text

iface = gr.Interface(fn=chat_with_model, inputs="text", outputs="text")
iface.launch()
