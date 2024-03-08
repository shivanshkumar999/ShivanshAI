from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn, torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI()

model_path = 'trained_chatbot_model_v2'

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def chatbot(prompt, model, tokenizer):
    prompt = "chatbot: " + prompt + " </s>"
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt') 

    with torch.no_grad():
        output = model.generate(prompt_ids, max_length=500, num_beams = 5)

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get('/',response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse('index.html',{"request":request})

@app.post('/chat')
async def chat(prompt:str=Form(...)):
    output = chatbot(prompt, model, tokenizer)
    return output

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)