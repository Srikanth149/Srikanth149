from pydantic import BaseModel
class Person(BaseModel):
    issue:str
    sentiment:str
    product:str
    logistics:str
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template(
    """
extract the follwoing fields from the input sentence:
-issue
-sentiment
-product
-logistics
return it as JSON
Sentence: {sentence}
"""
)

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
llm=ChatOpenAI(model='gpt-3.5-turbo')
parser = PydanticOutputParser(pydantic_object=Person)
chain = prompt | llm |parser

result = chain.invoke({"sentence" : "The customer complained about delayed delivery of his laptop"})
print(result)
print(result.issue)
print(result.sentiment)
print(result.product)
print(result.logistics)

