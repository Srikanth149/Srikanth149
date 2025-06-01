import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load BERT Q&A model
@st.cache_resource
def load_model():
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"  # Updated Model Name
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

# Streamlit UI
st.title("🤖 BERT-based Q&A Extractive Chatbot")
st.subheader("Ask questions based on a given context")

# User input: Context paragraph
context = st.text_area("Enter context paragraph:", 
                       placeholder="E.g., The sun rises in the east and sets in the west. The Earth rotates around the sun.")

# User input: Question
question = st.text_input("Ask a question:", placeholder="E.g., Where does the sun rise?")

if st.button("Get Answer"):
    if context.strip() and question.strip():
        with st.spinner("Thinking... 🤔"):
            result = qa_pipeline({"question": question, "context": context})
            answer = result["answer"]
            start= result["start"]
            end= result["end"]

            # Display answer
            st.success("✅ Answer:")
            st.write(f"**{answer}**")
            st.write(f"**{start}**")
            st.write(f"**{end}**")

            # Confidence Score
            st.write(f"🔍 Confidence Score: {result['score']:.2f}")

    else:
        st.error("❌ Please enter both context and question.")