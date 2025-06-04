from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Initialize the model
model = ChatOpenAI()

# Define dynamic chat prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Maintain chat history
chat_history = []

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    # Format the prompt using current chat history and user input
    messages = prompt.format_messages(
        chat_history=chat_history,
        input=user_input
    )

    # Call the model
    response = model.invoke(messages)

    # Save both user and AI messages into history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))

    print("AI:", response.content)
