from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
load_dotenv()
model1=ChatOpenAI()
model2=ChatOpenAI()
prompt1= PromptTemplate(
    template='Generate short and simple notes on text \n {text}',
    input_variables=['text']
)
prompt2 = PromptTemplate(
    template='Generate 3 short question and answers from the text \n {text}',
    input_variables=['text']
    )
prompt3 = PromptTemplate(
    template='Merge provided notes and quiz into a single document \n notes ->{notes},quiz -> {quiz}',
    input_variables=['notes','quiz']
)
Parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        'notes':prompt1| model1|Parser,
        'quiz':prompt2|model2|Parser
    }
)
merge_chain = prompt3|model1|Parser
chain= parallel_chain | merge_chain
text = """
An enhancement was implemented by the modeling team for a specific measure “Weighted Average Base Price Per Volume” in the QA environment, which was subsequently signed off and later deployed to production. However, during this timeline, an RC request was made on the STG environment. As a result, STG model files were pushed to UAT before the QA fix went live. This caused the UAT fix to be rolled back, although the fix did make it to production.
In a later build, the fix was missing because synchronization was not carried out between UAT and STG, leading to its absence in the most recent deployment.
"""
result = chain.invoke({'text':text})
print(result)