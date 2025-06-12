from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model_name="gpt-3.5-turbo")
parser = StrOutputParser()
class Feedback(BaseModel):
    sentiment: Literal['positive','negative'] = Field (description='give sentiment of the feedback')
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="classify the sentiment of the follwoing feedback text to positive or negative \n {feedback} \n {format_instruction}",
    input_variables= ['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template= "write an appropriate response to the postive feedback \n {feedback}",
    input_variables= ['feedback'])
prompt3 = PromptTemplate(
    template= "write an appropriate response to the negative feedback \n {feedback}",
    input_variables= ['feedback'])
branch_chain = RunnableBranch(
    (lambda x: x.sentiment =='positive',prompt2 |model|parser),
    (lambda x: x.sentiment =='negative',prompt3 |model|parser),
    RunnableLambda(lambda x: "couldnt find the sentiment")
)
chain =classifier_chain|branch_chain
user_input = input("Enter your feedback")
print(chain.invoke({'feedback': user_input}))

    