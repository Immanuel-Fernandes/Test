import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5-small model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(input_text):
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    st.title("Text Summarization App")

    input_text = st.text_area("Enter text to summarize:", height=200)
    
    if st.button("Summarize"):
        if input_text.strip() == "":
            st.error("Input text cannot be empty.")
        else:
            summary = summarize_text(input_text)
            st.subheader("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()
