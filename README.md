# Image-to-Text Application

An application that generates descriptive captions for images using a deep learning model. The app leverages Hugging Face’s **Salesforce/blip-image-captioning-large** model for image-to-text transformation and is deployed on Hugging Face Spaces using Gradio.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example Code](#example-code)
- [Deployment](#deployment)
- [Credits](#credits)

## Features
- **Image Captioning**: Generates captions that describe the contents of an image.
- **User-Friendly Interface**: Built with Gradio for an easy-to-use web interface.
- **Deployed on Hugging Face**: Accessible online for quick testing and demonstrations.

## Installation
To run this application locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
2. **Install Dependencies**: Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install -r requirements.txt

## Usage
1. **Run the application with**:
   ```bash
   python app.py

## Example Code:
1. Here’s a quick look at the core functionality:
   ```python
   from transformers import pipeline
   from PIL import Image
   import gradio as gr
   
   model = None
   
   if model == None:
       model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
   
   def captioner(image):
       img = Image.fromarray(image)
       result = model(img)[0]['generated_text']
       return result
   
   iface = gr.Interface(
       fn=captioner,
       inputs=gr.Image(),
       outputs='text'
   )
   
   iface.launch()

## Deployement:
- The app is deployed on Hugging Face Spaces. You can try it directly by uploading images to receive captions generated by the model.
