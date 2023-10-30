import streamlit as st
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    return processor, model

def predict(image, text, processor, model):
    encoding = processor(image, text, return_tensors="pt")
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]

def main():
    st.title("VQA")
    st.write("Upload an image and input a question to get an answer.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        question = st.text_input("Question about the image:")
        
        if question:
            processor, model = load_model()
            answer = predict(image, question, processor, model)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption='Uploaded Image.', use_column_width=True)
            with col2:
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
