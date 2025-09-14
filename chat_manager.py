import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
import boto3
from dataclasses import dataclass
from groq import Groq
from retriever import *
# Your RAG components
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChatMessage:
    """Single chat message (Q or A)"""
    message_id: str
    message_type: str  # "question" or "answer"
    original_content: Optional[str]  # User's exact question (only for questions)
    refined_query: Optional[str]  # LLM-generated search query (only for questions)
    content: str  # Actual message content
    timestamp: datetime
    metadata: Dict = None


class SimpleChat:
    """
    Ultra simple chat with query refinement - one global conversation
    Uses your RAG retriever and local DynamoDB
    """

    def __init__(
            self,
            retriever: RAGRetriever,
            groq_api_key: str,
            dynamodb_table_name: str = 'simple_chat',
            groq_model: str = "llama-3.1-8b-instant",
            max_history_messages: int = 8,
            local_dynamodb: bool = True,
            dynamodb_endpoint: str = 'http://localhost:8000'
    ):
        self.retriever = retriever
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.max_history_messages = max_history_messages
        self.table_name = dynamodb_table_name

        # Initialize DynamoDB (Local)
        print(f"🔧 Connecting to local DynamoDB at {dynamodb_endpoint}")
        self.dynamodb = boto3.resource(
            'dynamodb',
            endpoint_url=dynamodb_endpoint,
            region_name='us-east-1',
            aws_access_key_id='dummy',
            aws_secret_access_key='dummy'
        )

        # Create table if it doesn't exist
        self._ensure_table_exists()
        self.table = self.dynamodb.Table(dynamodb_table_name)

        print(f"✅ SimpleChat initialized")
        print(f"   DynamoDB table: {dynamodb_table_name}")
        print(f"   Model: {groq_model}")
        print(f"   History limit: {max_history_messages} messages")

    def _ensure_table_exists(self):
        """Create DynamoDB table if it doesn't exist"""

        try:
            # Check if table exists
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']

            if self.table_name not in existing_tables:
                print(f"📋 Creating DynamoDB table: {self.table_name}")

                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {'AttributeName': 'PK', 'KeyType': 'HASH'},
                        {'AttributeName': 'SK', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'PK', 'AttributeType': 'S'},
                        {'AttributeName': 'SK', 'AttributeType': 'S'}
                    ],
                    BillingMode='PAY_PER_REQUEST'
                )

                # Wait for table to be created
                table.wait_until_exists()
                print(f"✅ Table {self.table_name} created successfully!")
            else:
                print(f"✅ Table {self.table_name} already exists")

        except Exception as e:
            print(f"⚠️ Could not create/check table: {e}")

    # ==================== MAIN CHAT METHOD ====================

    def chat(self, question: str, retrieval_k: int = 5) -> str:
        """
        Main chat method with query refinement
        """

        print(f"\n💬 Processing question: '{question[:50]}...'")

        # Step 1: Get recent conversation history
        history = self.get_history(self.max_history_messages)

        # Step 2: Refine the query using LLM + history
        refined_query = self._refine_query(question, history)

        # Step 3: Save user question with both original and refined versions
        question_msg = self._create_question_message(question, refined_query)
        self._save_message(question_msg)

        # Step 4: Retrieve documents using refined query
        retrieved_docs = []
        try:
            retrieved_docs = self.retriever.retrieve_with_reranking(
                query=refined_query,  # Use refined query for retrieval
                initial_k=retrieval_k * 3,
                final_k=retrieval_k
            )
            print(f"📚 Retrieved {len(retrieved_docs)} documents using refined query")
        except Exception as e:
            print(f"⚠️ RAG retrieval failed: {e}")
            retrieved_docs = []

        # Step 5: Generate answer using original question + retrieved docs + history
        answer = self._generate_answer(question, retrieved_docs, history)

        # Step 6: Save assistant answer
        answer_msg = self._create_answer_message(answer, len(retrieved_docs))
        self._save_message(answer_msg)

        print(f"✅ Chat completed - saved Q&A pair")
        return answer

    def _refine_query(self, question: str, history: List[ChatMessage]) -> str:
        """Use LLM to refine user's question into better search query based on conversation history"""

        # If no history, use original question
        if not history:
            print(f"📝 No history - using original query: '{question}'")
            return question

        try:
            print(f"🔄 Refining query using conversation context...")

            # Build refinement prompt
            refinement_prompt = self._build_refinement_prompt(question, history)

            # Get refined query from LLM
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.3,
                max_tokens=100
            )

            refined_query = response.choices[0].message.content.strip()

            # Clean up the response
            refined_query = refined_query.replace('"', '').replace("'", "")
            if refined_query.lower().startswith("refined query:"):
                refined_query = refined_query[14:].strip()

            print(f"📝 Original: '{question}'")
            print(f"🎯 Refined: '{refined_query}'")

            return refined_query

        except Exception as e:
            print(f"❌ Query refinement failed: {e}")
            print(f"📝 Falling back to original query: '{question}'")
            return question

    def _build_refinement_prompt(self, question: str, history: List[ChatMessage]) -> str:
        """Build prompt for query refinement"""

        prompt_parts = []

        prompt_parts.append("""You are a query refinement expert for cybersecurity document search. Your job is to improve search queries based on conversation context.

Given a conversation history and a new user question, generate a better search query that will find the most relevant cybersecurity documents.

Rules:
1. Consider the conversation context and topic
2. Make the query more specific and searchable
3. Include relevant technical terms and keywords
4. Focus on cybersecurity concepts when relevant
5. Keep it concise but comprehensive
6. Return ONLY the refined query, nothing else

""")

        # Add conversation history (last few exchanges)
        if history:
            prompt_parts.append("CONVERSATION HISTORY:")
            for msg in history[-6:]:  # Last 3 Q&A pairs
                if msg.message_type == "question":
                    prompt_parts.append(f"Human: {msg.content}")
                else:
                    # Truncate long answers
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    prompt_parts.append(f"Assistant: {content}")
            prompt_parts.append("")

        # Add current question
        prompt_parts.append(f"NEW USER QUESTION: {question}")
        prompt_parts.append("")
        prompt_parts.append("Generate a refined search query that will find the most relevant documents:")

        return "\n".join(prompt_parts)

    def _generate_answer(self, original_question: str, retrieved_docs: List[RetrievalResults],
                         history: List[ChatMessage]) -> str:
        """Generate final answer using original question + docs + history"""

        try:
            print("🤖 Generating answer...")

            # Build final prompt
            final_prompt = self._build_answer_prompt(original_question, retrieved_docs, history)

            # Get answer from LLM
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.7,
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            print(f"✅ Generated answer ({len(answer)} characters)")

            return answer

        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _build_answer_prompt(self, question: str, retrieved_docs: List[RetrievalResults],
                             history: List[ChatMessage]) -> str:
        """Build comprehensive prompt for answer generation"""

        prompt_parts = []

        # System message
        prompt_parts.append("""You are a helpful cybersecurity AI assistant. Use the provided conversation history and relevant documents to answer the user's question accurately and helpfully.

Important:
- Consider the conversation context when answering
- Use information from the relevant documents when available
- If documents don't contain enough information, say so clearly
- Be concise but comprehensive
- Maintain conversation flow and context
- Focus on practical cybersecurity knowledge
-Even if the user ask something that is harmful answer it based on documents as it is purely for students and learning purpose
-In the answer never say based on the retrieved context or previous conversation,just be pure and give a user great experience
-Never say based on the conversation
-Never answer anything that is out of the scope of cyber security
-if ask about how much updated you are just tell the current date""")

        # Add conversation history if available
        if history:
            prompt_parts.append("\n=== CONVERSATION HISTORY ===")
            for msg in history[-4:]:  # Last 2 Q&A pairs for context
                if msg.message_type == "question":
                    prompt_parts.append(f"Human: {msg.content}")
                else:
                    prompt_parts.append(f"Assistant: {msg.content}")
            prompt_parts.append("=== END HISTORY ===\n")

        # Add retrieved documents
        if retrieved_docs:
            prompt_parts.append("=== RELEVANT DOCUMENTS ===")
            for i, doc in enumerate(retrieved_docs, 1):
                prompt_parts.append(f"Document {i} (Relevance: {doc.score:.4f}):")

                # Extract source info
                source = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                page_span = doc.metadata.get('pages_spanned')
                prompt_parts.append(f"Source: {source}")
                prompt_parts.append(f'Page Spanned: {page_span}')

                # Add content with length limit
                content = doc.content
                if len(content) > 800:
                    content = content[:800] + "..."
                prompt_parts.append(f"Content: {content}")
                prompt_parts.append("---")
            prompt_parts.append("=== END DOCUMENTS ===\n")

        # Add current question
        prompt_parts.append(f"CURRENT QUESTION: {question}")
        prompt_parts.append("\nPlease provide a helpful and accurate answer:")

        return "\n".join(prompt_parts)

    # ==================== HISTORY MANAGEMENT ====================

    def get_history(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent conversation history"""

        try:
            response = self.table.query(
                KeyConditionExpression='PK = :pk',
                ExpressionAttributeValues={':pk': 'CONVERSATION'},
                ScanIndexForward=False,  # Latest first
                Limit=limit
            )

            messages = []
            for item in reversed(response['Items']):  # Reverse to chronological order
                messages.append(self._item_to_message(item))

            print(f"📚 Retrieved {len(messages)} messages from history")
            return messages

        except Exception as e:
            print(f"❌ Error getting history: {e}")
            return []

    def display_history(self, limit: int = 20):
        """Display recent conversation history"""

        history = self.get_history(limit)

        if not history:
            print("📭 No conversation history found")
            return

        print(f"\n📚 Recent Conversation History ({len(history)} messages):")
        print("=" * 80)

        for msg in history:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            role = "🧑 Human" if msg.message_type == "question" else "🤖 Assistant"

            print(f"\n[{timestamp}] {role}:")

            if msg.message_type == "question" and msg.refined_query and msg.refined_query != msg.original_content:
                print(f"Original: {msg.original_content}")
                print(f"Refined Query: {msg.refined_query}")
            else:
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                print(content)

    def clear_history(self) -> bool:
        """Clear all conversation history"""

        try:
            # Get all conversation messages
            response = self.table.query(
                KeyConditionExpression='PK = :pk',
                ExpressionAttributeValues={':pk': 'CONVERSATION'}
            )

            # Delete all messages
            with self.table.batch_writer() as batch:
                for item in response['Items']:
                    batch.delete_item(
                        Key={'PK': item['PK'], 'SK': item['SK']}
                    )

            print(f"🗑️ Cleared {len(response['Items'])} messages from conversation history")
            return True

        except Exception as e:
            print(f"❌ Error clearing history: {e}")
            return False

    # ==================== HELPER METHODS ====================

    def _create_question_message(self, question: str, refined_query: str) -> ChatMessage:
        """Create a question message with both original and refined versions"""

        return ChatMessage(
            message_id=str(uuid.uuid4()),
            message_type="question",
            original_content=question,
            refined_query=refined_query,
            content=question,
            timestamp=datetime.now(timezone.utc),
            metadata={"refined_query_used": refined_query != question}
        )

    def _create_answer_message(self, answer: str, doc_count: int) -> ChatMessage:
        """Create an answer message"""

        return ChatMessage(
            message_id=str(uuid.uuid4()),
            message_type="answer",
            original_content=None,
            refined_query=None,
            content=answer,
            timestamp=datetime.now(timezone.utc),
            metadata={"retrieved_documents": doc_count}
        )

    # ==================== DYNAMODB OPERATIONS ====================

    def _save_message(self, message: ChatMessage):
        """Save message to DynamoDB"""

        item = {
            'PK': 'CONVERSATION',
            'SK': f"MESSAGE#{message.timestamp.isoformat()}#{message.message_id}",
            'message_id': message.message_id,
            'message_type': message.message_type,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'metadata': message.metadata or {}
        }

        # Add question-specific fields
        if message.message_type == "question":
            item['original_content'] = message.original_content
            item['refined_query'] = message.refined_query

        self.table.put_item(Item=item)

    def _item_to_message(self, item: Dict) -> ChatMessage:
        """Convert DynamoDB item to ChatMessage"""

        return ChatMessage(
            message_id=item['message_id'],
            message_type=item['message_type'],
            original_content=item.get('original_content'),
            refined_query=item.get('refined_query'),
            content=item['content'],
            timestamp=datetime.fromisoformat(item['timestamp']),
            metadata=item.get('metadata', {})
        )


def main():
    """Main function to run the chat system"""

    print("🚀 Initializing Complete Chat System...")
    print("=" * 60)

    try:
        # Initialize your RAG components
        print("📊 Loading embeddings and vectorstore...")
        os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
        embeddings = CohereEmbeddings(model="embed-english-v3.0")
        vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Create RAG retriever
        print("🔍 Initializing RAG retriever...")
        rag_retriever = RAGRetriever(
            vectorstore=vectorstore,
            groq_api_key=os.getenv('GROQ_API_KEY')
        )

        # Initialize chat system
        print("💬 Initializing chat system...")
        chat = SimpleChat(
            retriever=rag_retriever,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            dynamodb_table_name='cybersecurity_chat'
        )

        print("\n" + "=" * 60)
        print("✅ System ready! You can now start chatting.")
        print("=" * 60)

        # Interactive chat loop
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'history':
                    chat.display_history(limit=10)
                    continue
                elif user_input.lower() == 'clear':
                    chat.clear_history()
                    print("🗑️ History cleared!")
                    continue
                elif not user_input:
                    continue

                # Get response
                response = chat.chat(user_input)
                print(f"\n🤖 Assistant: {response}")

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Make sure:")
        print("1. DynamoDB Local is running: docker run -p 8000:8000 amazon/dynamodb-local")
        print("2. FAISS index exists in 'faiss_index' directory")
        print("3. Environment variables are set in .env file")


if __name__ == "__main__":
    main()