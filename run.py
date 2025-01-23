import io
import json
import requests
import streamlit as st
import torch
import PIL.Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
import re
import pytesseract
from PIL import Image
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk
import aspose.slides as slides
import aspose.pydrawing as drawing
from PyPDF2 import PdfReader
import os

nltk.download('punkt')

# Load pre-trained models for image captioning
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Function to extract images from PDF using PyPDF2 (modern library)
def extract_images_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        images = []
        index = 0
        for i, page in enumerate(reader.pages):
            print(i)
            # Assuming method for extracting images from page
            # You may need to use an external library or implement custom extraction logic here
            for image in page['/Resources']['/XObject'].values():
                if image['/Subtype'] == '/Image':
                    img_data = image.get_data()
                    image_file = f"Image-{index}.png"
                    with open(image_file, "wb") as img_f:
                        img_f.write(img_data)
                    images.append(image_file)
                    index += 1
        st.write("Total Extracted Images: ", len(images))
        return images
    except Exception as e:
        st.error(f"Error occurred while extracting images from PDF: {e}")
        return []

# Function to extract images from PowerPoint using Aspose
def extract_images_from_ppt(ppt_path):
    imageIndex = 1

    with slides.Presentation(ppt_path) as pres:
        for image in pres.images:
            image_type = image.content_type.split("/")[1].lower()
            file_name = f"Image_{imageIndex}.{image_type}"
            image_format = get_image_format(image_type)
            if image_format:
                image.system_image.save(file_name, image_format)
                imageIndex += 1
    st.write("Total Extracted Images: ", imageIndex - 1)
    return imageIndex - 1

# Function to get appropriate image format for saving
def get_image_format(image_type):
    return {
        "jpeg": drawing.imaging.ImageFormat.jpeg,
        "png": drawing.imaging.ImageFormat.png,
    }.get(image_type, None)

# Function to predict captions using the VisionEncoderDecoderModel
def predict_step(images):
    processed_images = []
    for image in images:
        resized_image = image.resize((224, 224))  # Resize to the expected input size
        np_image = np.array(resized_image)
        if np_image.shape[-1] != 3:
            np_image = np_image[..., :3]  # Ensure 3 channels (RGB)
        processed_images.append(np_image)

    processed_images = [PIL.Image.fromarray(img) for img in processed_images]

    pixel_values = processor(images=processed_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

# Streamlit UI setup
st.title("Image Captioning")

option = st.radio("Choose an option:", ("Extract images from PDF", "Extract images from PPT"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result = predict_step([image])
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("### Predicted Caption:")
        st.write(result[0])

elif option == "Extract images from PDF":
    pdf_file = st.file_uploader("Upload a PDF file...", type=["pdf"])

    if pdf_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        pdf_images = extract_images_from_pdf("temp.pdf")
        os.remove("temp.pdf")

        for img_path in pdf_images:
            image = PIL.Image.open(img_path)
            cap = predict_step([image])

            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
            extracted_text = pytesseract.image_to_string(image)

            clean_text = re.sub(r'[^\w\s]', '', extracted_text)
            clean_text = re.sub(r'\d+', '', clean_text)

            sentences = sent_tokenize(clean_text)
            
            nouns = []
            verbs = []
            adjectives = []
            if sentences:
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    try:
                        tagged_words = pos_tag(words)
                        for word, pos in tagged_words:
                            if pos.startswith('NN'):
                                nouns.append(word)
                            elif pos.startswith('VB'):
                                verbs.append(word)
                            elif pos.startswith('JJ'):
                                adjectives.append(word)

                        paragraph = f"The extracted text from the image contains {len(sentences)} sentences. " \
                                    f"It discusses topics like {', '.join(set(nouns))} and actions like " \
                                    f"{', '.join(set(verbs))}. The text also includes descriptive words like " \
                                    f"{', '.join(set(adjectives))}."

                        st.write(paragraph)
                    except Exception as e:
                        paragraph = "Error in processing sentence"
            else:
                paragraph = "No sentences extracted"

            st.image(image, caption=f'{cap} \n {paragraph}')

elif option == "Extract images from PPT":
    ppt_file = st.file_uploader("Upload a PPT file...", type=["pptx"])

    if ppt_file is not None:
        with open("temp.pptx", "wb") as f:
            f.write(ppt_file.getvalue())

        ppt_images = extract_images_from_ppt("temp.pptx")
        os.remove("temp.pptx")

        for i in range(ppt_images):
            try:
                image = PIL.Image.open(f'Image_{i+1}.jpeg')
            except Exception:
                image = PIL.Image.open(f'Image_{i+1}.png')

            cap = predict_step([image])

            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
            extracted_text = pytesseract.image_to_string(image)

            clean_text = re.sub(r'[^\w\s]', '', extracted_text)
            clean_text = re.sub(r'\d+', '', clean_text)

            sentences = sent_tokenize(clean_text)

            nouns = []
            verbs = []
            adjectives = []
            if sentences:
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    try:
                        tagged_words = pos_tag(words)
                        for word, pos in tagged_words:
                            if pos.startswith('NN'):
                                nouns.append(word)
                            elif pos.startswith('VB'):
                                verbs.append(word)
                            elif pos.startswith('JJ'):
                                adjectives.append(word)

                        paragraph = f"The extracted text from the image contains {len(sentences)} sentences. " \
                                    f"It discusses topics like {', '.join(set(nouns))} and actions like " \
                                    f"{', '.join(set(verbs))}. The text also includes descriptive words like " \
                                    f"{', '.join(set(adjectives))}."

                        st.write(paragraph)
                    except Exception as e:
                        paragraph = "Error in processing sentence"
            else:
                paragraph = "No sentences extracted"

            st.image(image, caption=f'{cap} \n {paragraph}')
