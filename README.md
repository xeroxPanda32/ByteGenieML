# Overview

This project is used for generating summary from text content. It generates summary from the provided text using ML model.

We will use Google Colab to run our ML model. The ML model is available on Hugging Face. 

> model used: facebook/bart-large-cnn (pre-trained)

# Project Setup
 
* Open new notebook on Google Colab https://colab.new/

* Install the dependencies mentioned in **requirements.txt** file in this repo on colab

   > pip install fastapi pyngrok uvicorn python-multipart kaleido nest_asyncio 
   
* To expose our server running on google colab to the internet, we will use ngrok **https://ngrok.com/**. Get auth token from ngrok
    >https://dashboard.ngrok.com/get-started/your-authtoken

* Set the ngrok authtoken in colab
   > !ngrok authtoken <--authtoken-->
* Copy the main code for running the ML model from **ml.py** into a code block on colab
   > Run the code

   You should see output similar to this 
   > FastAPI app is publicly accessible at: NgrokTunnel: "https://646f-34-90-50-191.ngrok-free.app" -> "http://localhost:8000"

   This public url will be used in backend application to make API calls to this ML server.

# APIs
The APIs are built using FastAPI.

* Server Check : API used for checking the availability of ML server

    > end point: "/check"

    > method: GET

    > parameters: None

    > return : JSON {"message":"Server online"}

* Generate Content : API used for generating content

    > end point: "/generateContent"

    > method: POST

    > parameters: req.body => JSON { "prompt": " paragraph that will be used for generating summary"}

    > return : JSON { "modelResponse": "Summary generated by ML" }