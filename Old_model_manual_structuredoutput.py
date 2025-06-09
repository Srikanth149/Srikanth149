from langchain_core.runnables import RunnableLambda
gpt_output= "product:smartphone-price:600"
def custome_parser(text:str)->dict:
    try:
        parts = text.split("-")
        product= parts[0].split(":")[1].strip()
        price=parts[1].split(":")[1].strip()
        return{"product":product,"price":price}
    except Exception as e:
        return{"error":str(e),"raw_output":text}
parser = RunnableLambda(custome_parser)
structured_output= parser.invoke(gpt_output)
print(structured_output)

""" When using older models like GPT-2, JSON output is not natively supported, 
so the response defaults to plain text, this requires manual parsing either by converting 
the text to a type dict or using tools like stroutputparse. In contrast, newer models like 
gpt 3.5,4 support structured outputs in formats like JSON which makes parsing more seamless by using Pydantic schemas.
"""