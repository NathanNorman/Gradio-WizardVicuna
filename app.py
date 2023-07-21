import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Make sure you have enough resources to load this model, as it's quite large
tokenizer = AutoTokenizer.from_pretrained("ehartford/WizardLM-30B-Uncensored")
model = AutoModelForCausalLM.from_pretrained("ehartford/WizardLM-30B-Uncensored")

def generate_text(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    text = tokenizer.decode(outputs[0])
    return text

iface = gr.Interface(fn=generate_text,
                     inputs=gr.inputs.Textbox(lines=2, placeholder='Enter a text prompt here...'),
                     outputs='text')

iface.launch()
