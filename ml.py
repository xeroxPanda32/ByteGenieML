from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nest_asyncio
import uvicorn

app = FastAPI()

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def generate_model_response(prompt):
    input_text = prompt

    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate output
    output_ids = model.generate(input_ids, max_length=200, num_beams=4, length_penalty=2.0, early_stopping=True)

    # Decode the generated output
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("Input text:", input_text)
    print("Generated output:", output_text)
    return output_text

@app.get('/check')
async def hello():
    return {"message": "Server online"}

@app.post('/generateContent')
async def generate_content(request: Request):
    try:
        data = await request.json()
        prompt = data.get('prompt')

        if prompt is None:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Perform model inference using the provided prompt
        model_response = generate_model_response(prompt)
        return JSONResponse(content={'modelResponse': model_response})

    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)

try:
    # Open a tunnel to the FastAPI app on port 8000
    public_url = ngrok.connect(8000)
    print('FastAPI app is publicly accessible at:', public_url)
except Exception as e:
    print('Error connecting with ngrok:', e)

nest_asyncio.apply()
uvicorn.run(app, port=8000)