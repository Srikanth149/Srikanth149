import streamlit as st
from transformers import AutoModelForCausalLM
st.title=("Llama-2 GGFU Chatbot")
st.subheader=("Chat with AI model local!")
@st.cache_resource
def load_model():
    model_path= "C:\Users\prvtg\OneDrive - IRI\Desktop\LLM\llama-2-7b-chat.Q4_K_M.gguf"
    model=AutoModelForCausalLM.from_pretrained(
        model_path,
        model_type="Llama",
        gpu_layer=0
        
    )
    return model
model=load_model()
user_input= st.text_area("enter your message:", placeholder="type something...")
if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("thinking"):
            response= model(user_input)
            st.success("AI Response")
            st.write(response)
    else:
        st.warning(" Warning! Enter some text before generating a response")

        
