from langchain_core.runnables import RunnableLambda, RunnableBranch 

# Define runnables
electronics = RunnableLambda(lambda x: f"electronics: {x.get('product_name', 'unknown product')}")
clothing = RunnableLambda(lambda x: f"clothing: {x.get('product_name', 'unknown product')}")
default = RunnableLambda(lambda x: f"default: {x.get('product_name', 'unknown product')}")

# Input
input_data = {"product_name": "Smartphone X100", "category": "electronics"}

# Routing
router = RunnableBranch(
    (lambda x: x.get("category", "").lower() == "electronics", electronics),
    (lambda x: x.get("category", "").lower() == "clothing", clothing),
    default
)

# Invoke and print
result = router.invoke(input_data)
print(result)
