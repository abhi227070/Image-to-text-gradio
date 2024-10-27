from transformers import pipeline
from PIL import Image
import gradio as gr

# Initialize the model
model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

def captioner(image):
    # Convert the image to PIL format
    img = Image.fromarray(image)
    # Generate the caption
    result = model(img)[0]['generated_text']
    return result

# Create Gradio Interface
iface = gr.Interface(
    fn=captioner,
    inputs=gr.Image(type="numpy"),
    outputs="text",
)

# Launch Gradio as an API server
iface.launch(server_name="0.0.0.0", server_port=8080)
