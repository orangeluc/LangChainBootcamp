from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = (
    ""
)

embeddings = OpenAIEmbeddings()


import arxiv
from langchain_community.retrievers import ArxivRetriever
from langchain_community.document_loaders import ArxivLoader, PDFMinerLoader
retriever = ArxivRetriever(load_max_docs=2)


# Create a search query
search = arxiv.Search(
    query="algo trading",
    max_results=2,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

def abs_to_pdf(abs_url):
    return abs_url.replace("abs", "pdf")
# get pdf id
pdf_ids = []
for result in search.results():
    docs = retriever.invoke(result.get_short_id())
    metadata = docs[0].metadata["Entry ID"]
    pdf_ids.append(metadata)

vdb_chunks  =  FAISS.from_documents([doc for doc in docs if doc.metadata["category"] != "Title"], embeddings=embeddings

docs=PDFMinerLoader(abs_to_pdf(pdf_ids[0])).load()
docs=text_splitter.split_documents(docs)

text_splitter = RecursiveCharacterTextSplitter()
docs = text_splitter.split_documents(docs)

vector = FAISS.from_documents(docs, embeddings)

retriever = vector.as_retriever()

retriever.get_relevant_documents("What are common pitfalls in using multi-objective optimization for algorithmic trading?")

