import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import warnings
from langchain import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights.*")

model_name = "sshleifer/distilbart-cnn-6-6"  # Using a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def summarize_text(input_text):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}\n\nSummary:",
    )
    llm = HuggingFacePipeline(pipeline=summarizer)
    summarization_chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        summary = summarization_chain.run({"text": input_text})
        return summary
    except Exception as e:
        return f"An error occurred: {e}"

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
