import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def summarize_text(input_text):
    parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return " ".join([str(sentence) for sentence in summary])

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
