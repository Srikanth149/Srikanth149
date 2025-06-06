
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain_core.runnables import RunnableLambda, RunnableBranch
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define intent labels
intent_labels = ["return", "complaint", "delivery", "general"]

# Step 2: Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Embed all intent labels
intent_embeddings = {
    label: embedding_model.embed_query(label)
    for label in intent_labels
}

# Step 4: Classifier using cosine similarity
def classify_intent(input_data):
    query = input_data["query"]
    query_embedding = embedding_model.embed_query(query)

    best_score = -1
    best_intent = "general"

    for label, emb in intent_embeddings.items():
        score = cosine_similarity([query_embedding], [emb])[0][0]
        if score > best_score:
            best_score = score
            best_intent = label

    return {"intent": best_intent, "query": query}

# Step 5: Wrap classifier in LangChain Runnable
intent_classifier = RunnableLambda(classify_intent)

# Step 6: Define intent logic branches
return_chain = RunnableLambda(lambda x: f"Return initiated: {x['query']}")
complaint_chain = RunnableLambda(lambda x: f"Complaint registered: {x['query']}")
delivery_chain = RunnableLambda(lambda x: f"Delivery info: {x['query']}")
general_chain = RunnableLambda(lambda x: f"General help: {x['query']}")

# Step 7: Use RunnableBranch
router = RunnableBranch(
    (lambda x: x["intent"] == "return", return_chain),
    (lambda x: x["intent"] == "complaint", complaint_chain),
    (lambda x: x["intent"] == "delivery", delivery_chain),
    general_chain
)

# Step 8: Combine into full chain
full_chain = intent_classifier | router

# Step 9: Interactive chatbot loop
if __name__ == "__main__":
    print("Chatbot ready! Type your query (type 'exit' to quit):")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Bot: Goodbye!")
            break
        result = full_chain.invoke({"query": user_input})
        print(f"Bot: {result}")
