from llama_index.core.indices.query.schema import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

from typing import List


class NovaceneRetriever(BaseRetriever):
    """Custom retriever that performs both Vector search and Knowledge Graph search"""
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        kg_retriever: KGTableRetriever,
        mode: str = "OR"
    ) -> None:
        """Init params."""
        
        self._vector_retriever = vector_retriever
        self._kg_retriever = kg_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]: 
        """Retrieve nodes given query."""
        
        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        kg_nodes = self._kg_retriever.retrieve(query_bundle)

        vector_ids = {n.node.get_doc_id() for n in vector_nodes}
        kg_ids = {n.node.get_doc_id() for n in kg_nodes}
        
        combined_dict = {n.node.get_doc_id(): n for n in vector_nodes}
        combined_dict.update({n.node.get_doc_id(): n for n in kg_nodes})
        
        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(kg_ids)
        else:
            retrieve_ids = vector_ids.union(kg_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes