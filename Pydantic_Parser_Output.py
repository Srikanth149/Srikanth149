from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel , Field
from langchain_openai import ChatOpenAI
model = ChatOpenAI()
class person(BaseModel):
    name:str=Field(description='name of the person')
    age:int=Field(description='age of the person')
    city:str=Field(description='name of the city he belongs to')
parser = PydanticOutputParser(pydantic_object=person)
tempalte = PromptTemplate(
    template= 'generate the name,age and city of the fictional person and the city must be ' \
    'exactly {place}\n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()
    }
)   
chain = tempalte|model|parser
result = chain.invoke({'place':'srilanka'})
print(result)