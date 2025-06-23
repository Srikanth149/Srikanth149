import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_thoughts(prompt, n=1):
    thoughts = []
    for i in range(n):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"{prompt}\nThought {i+1}:"}],
            temperature=0.7,
            max_tokens=150
        )
        thoughts.append(response.choices[0].message.content.strip())
    return thoughts



def run_tree_of_thoughts():
    prompt = """A retail company is facing issues with both overstocking and stockouts in its stores.  
Think step-by-step and propose a strategy to optimize inventory levels using demand forecasting and supply chain data.

Step 1: Identify the key causes of inventory imbalance in retail.
Step 2: Analyze what data is needed to forecast demand accurately.
Step 3: Suggest techniques/models to predict demand at SKU and store level.
Step 4: Recommend how to adjust inventory planning based on forecasts.
Step 5: Propose a feedback mechanism to continually improve accuracy.

Answer:"""

    thoughts = generate_thoughts(prompt)
    print("Generated Strategies:\n")
    for i, t in enumerate(thoughts):
        print(f"Thought {i+1}: {t}\n")


run_tree_of_thoughts()
