import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load a model specifically trained for English-to-Hindi translation
@st.cache_resource
def load_model():
    model_int = "Helsinki-NLP/opus-mt-en-hi"  # Use the best model for English-to-Hindi translation
    token= AutoTokenizer.from_pretrained(model_int)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_int)
    return token, model

token, model_int = load_model()

# Streamlit UI
st.title("AI-Powered English to Hindi Translator")
st.subheader("Translate English sentences into Hindi")

# User Input
input_text = st.text_area("Enter English Text:", placeholder="E.g., Come in", height=100)

if st.button("Translate to Hindi"):
    if input_text.strip():
        with st.spinner("Translating... Please wait ⏳"):
            #  Direct Translation (without additional formatting)
            input_ids = token.encode(input_text, return_tensors="pt")

            # Updated generation parameters for accurate translation
            output_ids = model_int.generate(
                input_ids,
                max_length=50,
                num_return_sequences=1,
                do_sample=False,  # Ensure deterministic translation
                temperature=0.7,
                repetition_penalty=1.5
            )
            translated_text = token.decode(output_ids[0], skip_special_tokens=True)

            # Display Result
            st.success("Translation Successful!")
            st.write("**Translated Hindi Text:**")
            st.write(translated_text)

            # Add Download Button
            st.download_button("📥 Download Translation", translated_text, file_name="translated_text.txt")
    else:
        st.error(" Please enter text before translating.")