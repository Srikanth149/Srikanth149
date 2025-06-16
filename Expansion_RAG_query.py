
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

# -------------------- Phase 1: Build and Save FAISS Index --------------------

def build_and_save_faiss_index():
    retail_chunks = [
        "Loyalty programs encourage repeat purchases and build long-term customer relationships.",
        "Churn can be reduced by identifying at-risk customers through predictive analytics.",
        "Retailers should segment customers and tailor marketing strategies to each segment.",
        "Offering personalized discounts based on purchase history increases retention.",
        "Customer feedback should be collected regularly to improve service and reduce dissatisfaction.",
        "Omnichannel support helps maintain a consistent customer experience across platforms.",
        "Surprise rewards or exclusive early access can increase customer engagement.",
        "Automated email campaigns triggered by inactivity can help win back customers.",
        "High churn is often linked to poor onboarding and unclear value communication.",
        "Customer education and clear value propositions lead to higher brand loyalty."
    ]

    documents = [Document(page_content=chunk) for chunk in retail_chunks]
    embedding_model = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(documents, embedding_model)
    vector_db.save_local("faiss_index")


# -------------------- Phase 2: Query Expansion & Retrieval QA --------------------

def load_vector_store():
    embedding_model = OpenAIEmbeddings()
    return FAISS.load_local(
        "faiss_index", 
        embedding_model, 
        allow_dangerous_deserialization=True
    )

def expand_query(llm, query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Generate 3 semantically different versions of the following query to improve search:\n\nQuery: {query}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(query)
    return [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

def rag_with_expansion(query):
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    vector_db = load_vector_store()
    queries = [query] + expand_query(llm, query)
    all_docs = []

    for q in queries:
        docs = vector_db.similarity_search(q, k=4)
        all_docs.extend(docs)

    # Remove duplicates
    unique_docs = list({doc.page_content: doc for e in all_docs}.values())
    context = "\n\n".join(doc.page_content for doc in unique_docs)

    answer_prompt = f"""You are an expert assistant. Use the following context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    return llm.predict(answer_prompt)

# -------------------- Run Entire Pipeline --------------------

if __name__ == "__main__":
    if not os.path.exists("faiss_index"):
        print("Index not found. Building FAISS index...")
        build_and_save_faiss_index()

    user_question = "How do I reduce churn in retail customers?"
    final_output = rag_with_expansion(user_question)

    print("Final Answer:\n")
    print(final_output)
