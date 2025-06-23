import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 1: Generate multiple strategies (thoughts)
def generate_thoughts(prompt, n=3):
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

# Step 2: Evaluate each strategy
def evaluate_thoughts(thoughts):
    evaluations = []
    for i, thought in enumerate(thoughts):
        eval_prompt = f"""Evaluate the effectiveness, cost, and customer impact of the following churn reduction strategy in retail:
Strategy: {thought}
Give a score from 1 to 10 and explain briefly."""
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.5,
            max_tokens=100
        )
        evaluations.append(response.choices[0].message.content.strip())
    return evaluations

# Step 3: Run the full Tree of Thoughts process
def run_tree_of_thoughts():
    prompt = "Suggest a strategy to reduce customer churn in a retail loyalty program."
    thoughts = generate_thoughts(prompt)
    print("Generated Strategies:\n")
    for i, t in enumerate(thoughts):
        print(f"Thought {i+1}: {t}\n")

    evaluations = evaluate_thoughts(thoughts)
    print("\nEvaluations:\n")
    for i, e in enumerate(evaluations):
        print(f"Evaluation {i+1}: {e}\n")

run_tree_of_thoughts()
