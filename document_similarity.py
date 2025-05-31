from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embedding= OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)
documents= ["You need to work independently on modeling tasks and proactively take the initiative before asked to address them",
"You need to focus on production issues to keep them to closure, identify the RCA if there are any slips and correct them proactively before the user raises them and ensure no issues in UAT environment upon sending the build to reduce the QA issues",
"You may need to review the automation code written by SK for MR and Panel Autobahn as this is crucial to help you grasp the concepts quickly. Please keep us updated on your progress with learning the automation code every week",
"Enhance time management and productivity by optimizing working hours and ensuring consistent focus during core work periods",
"Demonstrating strong skills in scripting to efficiently automate tasks, troubleshoot issues, and optimize workflows",
"Enhance clarity, responsiveness, and professionalism in all client interactions to build stronger relationships and ensure smooth collaboration"
]
query = "summarize on the automation task to enhance his perform better"
doc_embedding= embedding.embed_documents(documents)
query_embedding= embedding.embed_query(query)
scores = cosine_similarity([query_embedding],doc_embedding)[0]
index,score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(query)
print(documents[index])
print("similarity score is", score)


