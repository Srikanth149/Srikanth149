# 
# Multiple Chain of thoughts - > Tree of Thoughts
# import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 2. Country list
countries = ["USA", "New Zealand", "Australia", "Europe"]

# 3. Base step-by-step CoT prompt for each country
base_prompt = """You are a global retail pricing analyst.

A product is sold in USA, New Zealand, Australia, and Europe.
Based on current currency trends and market conditions, suggest a pricing strategy for one country.

Step 1: Analyze recent currency fluctuations vs USD.
Step 2: Assess local demand and pricing sensitivity.
Step 3: Compare with local competitor prices.
Step 4: Propose adjusted price and justification.
Step 5: Estimate expected impact on conversion and revenue.

Country: {COUNTRY}

Answer:"""

# 4. Generate one CoT per country
def generate_thoughts(base_prompt, countries):
    thoughts = []
    for country in countries:
        print(f"\n Prompting GPT for: {country}")
        prompt = base_prompt.format(COUNTRY=country)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )

        strategy = response.choices[0].message.content.strip()
        print(f"Strategy for {country}:\n{strategy}\n{'='*70}")
        thoughts.append((country, strategy))
    return thoughts

#  5. GPT evaluates all thoughts and picks the best
def gpt_select_best_strategy(thoughts):
    # Format all strategies into one block of text
    formatted_strategies = ""
    for country, strategy in thoughts:
        formatted_strategies += f"Country: {country}\nStrategy:\n{strategy}\n{'-'*50}\n"

    # GPT prompt to select the best strategy
    eval_prompt = f"""You are an expert pricing strategist.

You are given pricing strategies for four countries. Based on logic, effectiveness, and completeness, pick the BEST strategy and explain why.

{formatted_strategies}

Respond ONLY with:
Best Country: <country name>
Reason: <short reason>
"""

    print("\n Sending all strategies to GPT for evaluation...\n")
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.3,
        max_tokens=300,
    )

    result = response.choices[0].message.content.strip()
    print(f"GPT Evaluation Result:\n{result}\n{'='*70}")

    # Parse GPT's answer to find the selected country
    for country, strategy in thoughts:
        if country.lower() in result.lower():
            return country, strategy

    print("GPT did not clearly specify a known country. Defaulting to first.")
    return thoughts[0]

# 6. Main function to run the full ToT flow
def run_tree_of_thoughts():
    print("\n Generating pricing strategies for all countries...\n")
    thoughts = generate_thoughts(base_prompt, countries)

    print("\n GPT will now select the best pricing strategy...\n")
    best_country, best_strategy = gpt_select_best_strategy(thoughts)

    print("\n FINAL SELECTED STRATEGY:\n")
    print(f"Best Country: {best_country}\n\n{best_strategy}\n")

# 7. Run the code
if __name__ == "__main__":
    run_tree_of_thoughts()
