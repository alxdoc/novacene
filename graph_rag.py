from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import KGTableRetriever



from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq

from llama_index.core import Settings

from llama_index.embeddings.huggingface_optimum import OptimumEmbedding

from novacene_retriever import NovaceneRetriever

#Будем использовать отимизированную onnix модель e5-base-en-ru или есть -large вариант, так же есть эмбединги от банка Точка - чуть лучше
# чтобы сконвертировать в onnix раскомментируйте эти строки, появится папка с моделью и тогда надо обратно закомментировать
#OptimumEmbedding.create_and_save_optimum_model(
#    "d0rj/e5-base-en-ru", "./e5-base-en-ru"
#)

# Можно использовать локальную Llama3 через Ollama
#Settings.llm = Ollama(model="llama3", request_timeout=90.0)
# Либо Groq - они дают доступ бесплатно, но работает только за VPN
llm = Groq(model="llama3-8b-8192", api_key="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") #llama3-70b-8192 llama3-8b-8192
Settings.llm = llm
#Settings.embed_model = HuggingFaceEmbedding(model_name="d0rj/e5-base-en-ru")
Settings.embed_model = OptimumEmbedding(folder_name="./e5-base-en-ru")

# Загрузка документов
#TODO тут код загрузки документов в vector store и в graph store
# Этот код убрать - берется статья из википедии и на ее основе создаются документы
# в нашем случае это карточки с вопросами и ответами и мета-информацией
wikipedia_reader = WikipediaReader()
documents = wikipedia_reader.load_data(pages=['Россия'], auto_suggest=False)
for doc in documents:
    print(doc)

#TODO заменить на это !!!
#space_name = "novacene_rag"
#edge_types, rel_prop_names = ["relationship"], ["relationship"]
#tags = ["entity"]

#graph_store = NebulaGraphStore(
#    space_name=space_name,
#    edge_types=edge_types,
#    rel_prop_names=rel_prop_names,
#    tags=tags,
#)
#storage_context = StorageContext.from_defaults(graph_store=graph_store)

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=2,
    storage_context=storage_context,
)

#kg_index = KnowledgeGraphIndex.from_documents(
#    documents,
#    storage_context=storage_context,
#    service_context=service_context,
#    max_triplets_per_chunk=10,
#    space_name=space_name,
#    edge_types=edge_types,
#    rel_prop_names=rel_prop_names,
#    tags=tags,
#)

# create custom retriever
vector_retriever = VectorIndexRetriever(index=vector_index)
kg_retriever = KGTableRetriever(index=kg_index, retriever_mode='keyword', include_text=False)
novacene_retriever = NovaceneRetriever(vector_retriever, kg_retriever)

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize",)
custom_query_engine = RetrieverQueryEngine(
    retriever=novacene_retriever,
    response_synthesizer=response_synthesizer,
)

response = custom_query_engine.query(
    "Tell me more about Russia. Answer only in Russian.",
)

print('Response: ')
print(response)

