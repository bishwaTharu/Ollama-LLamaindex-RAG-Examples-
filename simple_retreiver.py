from llama_index.llms import Ollama
import qdrant_client
from llama_index import (
    VectorStoreIndex,
    ServiceContext
)
from llama_index.llms import Ollama
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

BASEURL = "https://03f8-35-243-188-201.ngrok-free.app"

llm = Ollama(model="llama2", request_timeout=30.0,base_url=BASEURL)

loader = PyMuPDFReader()

documents = loader.load(file_path="./machinelearning-lecture01.pdf")
client = qdrant_client.QdrantClient(
    path="./qdrant_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="mydata")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
service_context = ServiceContext.from_defaults(llm=llm,embed_model="local")

index = VectorStoreIndex.from_documents(documents,service_context=service_context,storage_context=storage_context)


query_engine = index.as_query_engine(streaming=True)
while True:
    try:
        question = input("Enter your query: ")
        response = query_engine.query(question)
        response.print_response_stream()
        print("\n")
    except KeyboardInterrupt:
        break