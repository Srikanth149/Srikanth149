from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
load_dotenv()
st.header("Region Sales for Speakers")
model = ChatOpenAI()
Region = st.selectbox("select Region",["APAC","EMEA","US","LATAM"])
State = st.selectbox("select State" ,["IND","ITL","US","CAN"])
template = PromptTemplate(
    template = """ Provide the {region}  
    GDP for the {state}
"""
,
input_variables=['region','state']
)
prompt = template.invoke(
    {'region':Region,
     'state':State
     }
)

if st.button ('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)