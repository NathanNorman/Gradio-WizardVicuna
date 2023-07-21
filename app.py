import gradio as gr
from transformers import AutoTokenizer, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Load pre-trained model and tokenizer
model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order"
use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    model_basename=model_basename,
    use_safetensors=True,
    trust_remote_code=False,
    device="cuda:0",
    use_triton=use_triton,
    quantize_config=None
)

def chat(input_text: str) -> str:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').cuda()

    # generate a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last output tokens from bot
    response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

iface = gr.Interface(
    fn=chat,  # function that processes the input and gives the output
    inputs=gr.inputs.Textbox(lines=3, placeholder='Enter text here...'),  # input component
    outputs=gr.outputs.Textbox()  # output component
)

iface.launch()
