from importlib.metadata import metadata

from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
import os
from typing import List,Dict,Tuple,Any
from dataclasses import dataclass
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import CrossEncoder


load_dotenv()

os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

@dataclass
class RetrievalResults:
    content: str
    metadata: Dict
    score: float

class RAGRetriever:
    def __init__(self,vectorstore: FAISS,groq_api_key: str,groq_model: str = "mixtral-8x7b-32768" ,cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.vectorstore = vectorstore
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        self.cross_encoder_model = cross_encoder_model
        print("✅ RAG Retriever Step 1 initialized!")

    def basic_retrieve(self, query: str,k: int = 10):
        print(f"🔍 Basic retrieval for: '{query}'")
        print(f"📝 Retrieving top {k} documents...")

        try:
            doc_with_score = self.vectorstore.similarity_search_with_score(query,k=k)

            results = []

            for doc,score in doc_with_score:
                result = RetrievalResults(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=float(score),
                )
                results.append(result)
            print(f"✅ Retrieved {len(results)} documents")
            return results

        except Exception as e:
            print("Error During retrieval",e)
            return []


    def cross_encode_rerank(self,query: str, results: List[RetrievalResults],top_k: int = 5) -> List[RetrievalResults]:
        if len(results)==0:
            print("No Results to re-reank")

        if len(results) <= 1:
            print("ℹ️ Only 1 result, no re-ranking needed")
            return results
        print(f"🔄 Re-ranking {len(results)} results using cross-encoder...")

        try:
            query_doc_pair = []
            for result in results:
                query_doc_pair.append([query,result.content])

            cross_scores = self.cross_encoder.predict(query_doc_pair)

            reranked_results = []

            for result,cross_scores in zip(results,cross_scores):
                reranked_result = RetrievalResults(
                    content=result.content,
                    metadata=result.metadata,
                    score=float(cross_scores)
                )
                reranked_results.append(reranked_result)

            reranked_results.sort(key=lambda x: x.score, reverse=True)

            final_results = reranked_results[:top_k]
            print(f"✅ Re-ranking complete! Returning top {len(final_results)} results")
            print(f"📈 Score range: {final_results[0].score:.4f} to {final_results[-1].score:.4f}")

            return final_results
        except Exception as e:
            print(f"❌ Error during cross-encoder re-ranking: {e}")
            print("🔄 Falling back to original results...")
            return results[:top_k]

    def retrieve_with_reranking(self,query: str, initial_k:int = 20,final_k:int=5)->List[RetrievalResults]:
        print(f"\n🎯 Starting retrieval with re-ranking for: '{query}'")
        initial_results = self.basic_retrieve(query,initial_k)
        if not initial_results:
            return []
        final_results = self.cross_encode_rerank(query,initial_results,top_k=final_k)
        print(f"🎉 Pipeline complete! {len(final_results)} final results")
        return final_results



    def display_results(self,results: List[RetrievalResults],max_content_length: int = 600):
        if not results:
            print("❌ No results to display")
            return

        print(f"\n📋 Displaying {len(results)} results:")
        print("=" * 80)

        for i, result in enumerate(results,1):
            content_preview = result.content[:max_content_length]
            if len(result.content) > max_content_length:
                content_preview += "..."
            print(f"Score of Retrieved Document: {result.score}")
            print(f"Content: {content_preview}")
            print(f"Metadata: {result.metadata}")



if __name__ == "__main__":
    # Your existing setup
    os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    retriever = RAGRetriever(
        vectorstore= vectorstore,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )


    query = "What is environment steup"
    results = retriever.retrieve_with_reranking(query,initial_k=10,final_k=5)

    retriever.display_results(results)



