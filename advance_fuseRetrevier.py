import asyncio
from typing import List
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
)

# importing module
import warnings

from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import PromptTemplate
from tqdm.asyncio import tqdm
from llama_index.retrievers import BM25Retriever
from llama_index.schema import NodeWithScore
from llama_index import QueryBundle
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import Ollama
from llama_hub.file.pymu_pdf.base import PyMuPDFReader

warnings.filterwarnings("ignore")

BASEURL = "https://4cc4-34-73-100-122.ngrok-free.app"

llm = Ollama(model="llama2", base_url=BASEURL)

loader = PyMuPDFReader()

documents = loader.load(file_path="./data/llama2.pdf")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, chunk_size=1024
)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)


query_str = "How do the models developed in this work compare to open-source chat models based on the benchmarks tested?"

query_gen_prompt_str = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)
query_gen_prompt = PromptTemplate(query_gen_prompt_str)


def generate_queries(llm, query_str: str, num_queries: int = 4):
    fmt_prompt = query_gen_prompt.format(num_queries=num_queries - 1, query=query_str)
    response = llm.complete(fmt_prompt)
    queries = response.text.split("\n")
    return queries


queries = generate_queries(llm, query_str, num_queries=4)
queries = [item for item in queries if item != ""]


async def run_queries(queries, retrievers):
    """Run queries against retrievers."""
    tasks = []
    for query in queries:
        for i, retriever in enumerate(retrievers):
            tasks.append(retriever.aretrieve(query))

    task_results = await tqdm.gather(*tasks)

    results_dict = {}
    for i, (query, query_result) in enumerate(zip(queries, task_results)):
        results_dict[(query, i)] = query_result

    return results_dict


## vector retriever
vector_retriever = index.as_retriever(similarity_top_k=2)

## bm25 retriever
bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore, similarity_top_k=2
)

results_dict = asyncio.run(run_queries(queries, [vector_retriever, bm25_retriever]))
# results_dict = await run_queries(queries, )


def fuse_results(results_dict, similarity_top_k: int = 2):
    """Fuse results."""
    k = 60.0  # `k` is a parameter used to control the impact of outlier rankings.
    fused_scores = {}
    text_to_node = {}

    # compute reciprocal rank scores
    for nodes_with_scores in results_dict.values():
        for rank, node_with_score in enumerate(
            sorted(nodes_with_scores, key=lambda x: x.score or 0.0, reverse=True)
        ):
            text = node_with_score.node.get_content()
            text_to_node[text] = node_with_score
            if text not in fused_scores:
                fused_scores[text] = 0.0
            fused_scores[text] += 1.0 / (rank + k)

    # sort results
    reranked_results = dict(
        sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    )

    # adjust node scores
    reranked_nodes: List[NodeWithScore] = []
    for text, score in reranked_results.items():
        reranked_nodes.append(text_to_node[text])
        reranked_nodes[-1].score = score

    return reranked_nodes[:similarity_top_k]


final_results = fuse_results(results_dict)


class FusionRetriever(BaseRetriever):
    """Ensemble retriever with fusion."""

    def __init__(
        self,
        llm,
        retrievers: List[BaseRetriever],
        similarity_top_k: int = 2,
    ) -> None:
        """Init params."""
        self._retrievers = retrievers
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        queries = generate_queries(llm, query_str, num_queries=4)
        results = run_queries(queries, [vector_retriever, bm25_retriever])
        final_results = fuse_results(
            results_dict, similarity_top_k=self._similarity_top_k
        )

        return final_results


fusion_retriever = FusionRetriever(
    llm, [vector_retriever, bm25_retriever], similarity_top_k=2
)


query_engine_base = RetrieverQueryEngine.from_args(
    fusion_retriever, service_context=service_context, streaming=True
)

while True:
    try:
        print("User:\n")
        query_inp = input()
        response = query_engine_base.query(query_inp)
        if query_inp == "/exit":
            break
        else:
            print("Assistant:\n")
            response.print_response_stream()
            print("\n\n")

    except KeyboardInterrupt:
        break
