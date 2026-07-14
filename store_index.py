from dotenv import load_dotenv
import os
import time
from src.helper import load_pdf_file, filter_to_doc, text_split, download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

extracted_data=load_pdf_file(data='data/')
filter_data = filter_to_doc(extracted_data)
text_chunks=text_split(filter_data)

print(f"Total chunks to embed: {len(text_chunks)}")

embeddings = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print("Waiting for index to be ready...")
    time.sleep(15)

index = pc.Index(index_name)

# Create the vector store from the first batch
BATCH_SIZE = 200
total = len(text_chunks)

print(f"Processing batch 1/{(total + BATCH_SIZE - 1) // BATCH_SIZE} (chunks 1-{min(BATCH_SIZE, total)})...")
docSearch = PineconeVectorStore.from_documents(
    documents=text_chunks[:BATCH_SIZE],
    index_name=index_name,
    embedding=embeddings,
)

# Add remaining chunks in batches without delays
for i in range(BATCH_SIZE, total, BATCH_SIZE):
    batch_num = (i // BATCH_SIZE) + 1
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    end = min(i + BATCH_SIZE, total)
    print(f"Processing batch {batch_num}/{total_batches} (chunks {i+1}-{end})...")
    docSearch.add_documents(text_chunks[i:end])

print(f"\nDone! Successfully indexed {total} chunks into Pinecone.")