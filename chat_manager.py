import os
import json
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import boto3
from dataclasses import dataclass
from groq import Groq
from retriever import *
# Your RAG components
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from enum import Enum

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
class SessionStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXPIRED = "expired"
    ENDED = "ended"

class MessageType(Enum):
    QUESTION = "question"
    ANSWER = "answer"
    SYSTEM = "system"
# ==================== DATA MODELS ====================

@dataclass
class User:
    user_id :str
    user_name:str
    email:str
    created_at :datetime
    last_active:datetime
    total_chatrooms:int=0
    metadata:Dict=None

@dataclass
class Chatroom:
    chatroom_id:str
    user_id:str
    chatroom_name:str
    description:str
    created_at: datetime
    last_active: datetime
    total_sessions: int = 0
    active_sessions: int = 0
    metadata: Dict = None

@dataclass
class Session:
    """Session entity - active conversation instance"""
    session_id: str
    chatroom_id: str
    user_id: str
    session_name: str
    status: SessionStatus
    created_at: datetime
    last_active: datetime
    expires_at: Optional[datetime]
    message_count: int = 0
    context_summary: str = ""
    metadata: Dict = None



@dataclass
class ChatMessage:
    """Enhanced message with full context"""
    message_id: str
    session_id: str
    chatroom_id: str
    user_id: str
    message_type: MessageType
    original_content: Optional[str]  # Original user question
    refined_query: Optional[str]  # AI-refined search query
    content: str  # Final content
    timestamp: datetime
    retrieved_docs_count: int = 0
    metadata: Dict = None

# ==================== ENHANCED CHAT SYSTEM ====================



class EnhancedRAGChat:
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
            session_timeout_hours:int=24,
            max_history_messages: int = 10,
            local_dynamodb: bool = True,
            dynamodb_endpoint: str = 'http://localhost:8000'
    ):
        self.retriever = retriever
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model
        self.session_timeout_hours = session_timeout_hours
        self.max_history_messages = max_history_messages
        self.table_name = dynamodb_table_name

        # Prepare LangChain PromptTemplates
        self.refine_template = PromptTemplate(
            input_variables=["question", "history"],
            template=(
                "You are a query refinement expert for cybersecurity document search. "
                "Your job is to improve search queries based on conversation context.\n\n"
                "Rules:\n"
                "1. Consider the conversation context and topic\n"
                "2. Make the query more specific and searchable\n"
                "3. Include relevant technical terms and keywords\n"
                "4. Focus on cybersecurity concepts when relevant\n"
                "5. Keep it concise but comprehensive\n"
                "6. Return ONLY the refined query, nothing else\n\n"
                "CONVERSATION HISTORY (most recent first, may be empty):\n{history}\n\n"
                "NEW USER QUESTION: {question}\n\n"
                "Return the refined search query:"
            ),
        )

        self.answer_template = PromptTemplate(
            input_variables=["question", "history", "documents", "context_summary"],
            template=(
                "You are a helpful cybersecurity AI assistant. Use the provided conversation "
                "history and relevant documents to answer the user's question accurately and helpfully.\n\n"
                "Important:\n"
                "- Consider the conversation context when answering\n"
                "- Use information from the relevant documents when available\n"
                "-Always answer from the documents do not add it by yourself "
                "- If documents don't contain enough information, say so clearly\n"
                "- Be concise but comprehensive\n"
                "- Maintain conversation flow and context\n"
                "- Focus on practical cybersecurity knowledge\n"
                "- Never state that you're using retrieved context or prior conversation explicitly\n"
                "- Do NOT use phrases like 'based on the conversation', 'based on the documents provided', 'according to the context', or similar. Speak directly without referencing sources.\n"
                "- Keep scope within cybersecurity only\n"
                "- If asked how current you are, reply with today's date\n\n"
                "=== CONVERSATION CONTEXT SUMMARY ===\n{context_summary}\n=== END CONTEXT ===\n\n"
                "=== RECENT HISTORY ===\n{history}\n=== END HISTORY ===\n\n"
                "=== RELEVANT DOCUMENTS ===\n{documents}\n=== END DOCUMENTS ===\n\n"
                "CURRENT QUESTION: {question}\n\n"
                "Please provide a helpful and accurate answer:"
            ),
        )

        # Summarization PromptTemplate for rolling conversation memory
        self.summary_template = PromptTemplate(
            input_variables=["previous_summary", "recent_messages"],
            template=(
                "You maintain a rolling, concise summary of a cybersecurity Q&A session.\n"
                "Update the summary to capture key facts, the user's stated preferences or identity, and ongoing task context.\n"
                "- Keep it neutral, factual, and under 120 words.\n"
                "- Include the user's name if they state it.\n"
                "- Exclude greetings, fillers, and model meta-comments.\n\n"
                "PREVIOUS SUMMARY:\n{previous_summary}\n\n"
                "RECENT MESSAGES:\n{recent_messages}\n\n"
                "Return ONLY the updated summary."
            ),
        )

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
        try:
            existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']

            if self.table_name not in existing_tables:
                print(f"📋 Creating scalable DynamoDB table: {self.table_name}")

                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {'AttributeName': 'PK', 'KeyType': 'HASH'},
                        {'AttributeName': 'SK', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'PK', 'AttributeType': 'S'},
                        {'AttributeName': 'SK', 'AttributeType': 'S'},
                        {'AttributeName': 'user_id', 'AttributeType': 'S'},
                        {'AttributeName': 'chatroom_id', 'AttributeType': 'S'},
                        {'AttributeName': 'last_active', 'AttributeType': 'S'}
                    ],
                    GlobalSecondaryIndexes=[
                        {
                            'IndexName': 'UserIndex',
                            'KeySchema': [
                                {'AttributeName': 'user_id', 'KeyType': 'HASH'},
                                {'AttributeName': 'last_active', 'KeyType': 'RANGE'}
                            ],
                            'Projection': {'ProjectionType': 'ALL'},
                            'ProvisionedThroughput': {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                        },
                        {
                            'IndexName': 'ChatroomIndex',
                            'KeySchema': [
                                {'AttributeName': 'chatroom_id', 'KeyType': 'HASH'},
                                {'AttributeName': 'last_active', 'KeyType': 'RANGE'}
                            ],
                            'Projection': {'ProjectionType': 'ALL'},
                            'ProvisionedThroughput': {'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                        }
                    ],
                    BillingMode='PROVISIONED',
                    ProvisionedThroughput={'ReadCapacityUnits': 5, 'WriteCapacityUnits': 5}
                )

                table.wait_until_exists()
                print(f"✅ Scalable table {self.table_name} created with GSIs!")
            else:
                print(f"✅ Table {self.table_name} already exists")

        except Exception as e:
            print(f"⚠️ Could not create table: {e}")
        """Create DynamoDB table with GSIs for scalable access patterns"""

    # ==================== USER MANAGEMENT ====================

    def create_user(
            self,
            username:str,
            email:str,
            user_id:str=None
            )->str:
        if not user_id:
            user_id = str(uuid.uuid4())

        user = User(
            user_id,
            username,
            email,
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc),
            metadata={}
        )
        self._save_user(user)
        print(f"👤 Created user: {username} ({user_id})")
        return user_id

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            response = self.table.get_item(
                Key={'PK': f"USER#{user_id}", 'SK': 'PROFILE'}
            )
            if 'Item' in response:
                return self._item_to_user(response['Item'])
            return None
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None

    def list_user_chatrooms(self, user_id: str, limit: int = 50) -> List[Chatroom]:
        """List all chatrooms for a user"""
        try:
            response = self.table.query(
                IndexName='UserIndex',
                KeyConditionExpression='user_id = :user_id',
                FilterExpression='begins_with(SK, :sk)',
                ExpressionAttributeValues={
                    ':user_id': user_id,
                    ':sk': 'METADATA'
                },
                ScanIndexForward=False,
                Limit=limit
            )

            chatrooms = []
            for item in response['Items']:
                if 'chatroom_id' in item:
                    chatrooms.append(self._item_to_chatroom(item))

            return chatrooms
        except Exception as e:
            print(f"❌ Error listing chatrooms: {e}")
            return []

    # ==================== CHATROOM MANAGEMENT ====================
    def create_chatroom(self,user_id:str,chatroom_name:str,description:str=""):
        chatroom_id = str(uuid.uuid4())
        chatroom = Chatroom(
            chatroom_id=chatroom_id,
            user_id=user_id,
            chatroom_name=chatroom_name,
            description=description,
            created_at=datetime.now(timezone.utc),
            last_active = datetime.now(timezone.utc),
            metadata={}
        )
        self._save_chatroom(chatroom)
        self._increment_user_chatrooms(user_id)

        print(f"🏠 Created chatroom: {chatroom_name} ({chatroom_id}) for user {user_id}")
        return chatroom_id

    def get_chatroom(self,chatroom_id:str)->Optional[Chatroom]:
        """Get chatroom by ID"""
        try:
            response = self.table.get_item(
                Key={'PK': f"CHATROOM#{chatroom_id}", 'SK': 'METADATA'}
            )
            if 'Item' in response:
                return self._item_to_chatroom(response['Item'])
            return None
        except Exception as e:
            print(f"❌ Error getting chatroom: {e}")
            return None
        # ==================== SESSION MANAGEMENT ====================

    def create_session(self,chatroom_id:str,user_id:str,session_name:str,timeout_hours:int=None)->str:
        session_id = str(uuid.uuid4())
        timeout_hours = timeout_hours or self.session_timeout_hours
        session=Session(
            session_id,
            chatroom_id,
            user_id,
            session_name,
            status=SessionStatus.ACTIVE,
            created_at = datetime.now(timezone.utc),
            last_active= datetime.now(timezone.utc),
            expires_at= datetime.now(timezone.utc) + timedelta(hours=timeout_hours),
            metadata={}
        )
        self._save_session(session)
        self._increment_chatroom_sessions(chatroom_id)
        print(f"🆕 Created session: {session_name} ({session_id}) in chatroom {chatroom_id}")
        return session_id
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        try:
            response = self.table.get_item(
                Key={'PK': f"SESSION#{session_id}", 'SK': 'METADATA'}
            )
            if 'Item' in response:
                return self._item_to_session(response['Item'])
            return None
        except Exception as e:
            print(f"❌ Error getting session: {e}")
            return None

    def list_chatroom_sessions(self, chatroom_id: str) -> List[Session]:
        """List sessions in a chatroom"""
        try:
            response = self.table.query(
                IndexName='ChatroomIndex',
                KeyConditionExpression='chatroom_id = :chatroom_id',
                FilterExpression='begins_with(SK, :sk)',
                ExpressionAttributeValues={
                    ':chatroom_id': chatroom_id,
                    ':sk': 'METADATA'
                },
                ScanIndexForward=False
            )

            sessions = []
            for item in response['Items']:
                if 'session_id' in item:
                    sessions.append(self._item_to_session(item))

            return sessions
        except Exception as e:
            print(f"❌ Error listing sessions: {e}")
            return []

    # ==================== MAIN CHAT METHOD ====================

    def chat(self,session_id:str,user_id:str, question: str, retrieval_k: int = 5) -> str:
        """
        Main chat method with query refinement
        """
        print(f"\n💬 Chat in session {session_id}: '{question[:50]}...'")
        session = self.get_session(session_id)
        if not session:
            return "❌ Session not found"

        if session.status not in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            return "❌ Session is not active"

        # Check session hasn't expired
        if session.expires_at and datetime.now(timezone.utc) > session.expires_at:
            self._update_session_status(session_id, SessionStatus.EXPIRED)
            return "❌ Session has expired"

        # Resume session if paused
        if session.status == SessionStatus.PAUSED:
            self._update_session_status(session_id, SessionStatus.ACTIVE)

        #Get Conversation History
        history = self.get_session_messages(session_id,limit=self.max_history_messages)

        refined_query = self._refine_query(question, history)

        question_msg = self._create_message(
            session_id=session_id,
            chatroom_id = session.chatroom_id,
            user_id=user_id,
            message_type = MessageType.QUESTION,
            content = question,
            original_content=question,
            refined_query=refined_query

        )
        self._save_message(question_msg)

        # Update rolling summary with the new question
        try:
            summary_input_messages = history + [question_msg]
            updated_summary = self._summarize_history(summary_input_messages, session.context_summary or "")
            if updated_summary:
                self._update_session_context_summary(session_id, updated_summary)
                session.context_summary = updated_summary
        except Exception as _:
            pass


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

        answer = self._generate_answer(question, retrieved_docs, history,session.context_summary)
        answer_msg = self._create_message(
            session_id=session_id,
            chatroom_id = session.chatroom_id,
            user_id=user_id,
            message_type= MessageType.ANSWER,
            content=answer,
            retrieved_docs_count=len(retrieved_docs)

        )
        self._save_message(answer_msg)

        # Update rolling summary with the new answer
        try:
            history_after_answer = history + [question_msg, answer_msg]
            updated_summary = self._summarize_history(history_after_answer, session.context_summary or "")
            if updated_summary:
                self._update_session_context_summary(session_id, updated_summary)
        except Exception as _:
            pass
        # Update session activity
        self._update_session_activity(session_id)

        print(f"✅ Chat completed in session {session_id}")
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
        """Build prompt for query refinement using LangChain PromptTemplate"""

        history_text = "\n".join([
            f"{'Human' if msg.message_type == MessageType.QUESTION else 'Assistant'}: "
            f"{(msg.content[:150] + '...') if len(msg.content) > 150 else msg.content}"
            for msg in reversed(history[-6:])
        ]) if history else "(no recent messages)"

        return self.refine_template.format(question=question, history=history_text)

    def _generate_answer(self, question: str, retrieved_docs, history: List[ChatMessage],
                         context_summary: str = "") -> str:
        """Generate answer using question + docs + history + context"""
        try:
            final_prompt = self._build_answer_prompt(question, retrieved_docs, history, context_summary)

            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.7,
                max_tokens=1000
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    def _build_answer_prompt(self, question: str, retrieved_docs, history: List[ChatMessage],
                             context_summary: str) -> str:
        """Build comprehensive answer generation prompt using LangChain PromptTemplate"""
        history_text = "\n".join([
            f"{'Human' if msg.message_type == MessageType.QUESTION else 'Assistant'}: {msg.content}"
            for msg in history[-4:]
        ]) if history else "(no recent messages)"

        if retrieved_docs:
            docs_text_lines = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get('title', doc.metadata.get('source', 'Unknown'))
                page_span = doc.metadata.get('pages_spanned')
                content = doc.content[:800] + "..." if len(doc.content) > 800 else doc.content
                docs_text_lines.extend([
                    f"Document {i} (Relevance: {doc.score:.4f}):",
                    f"Source: {source}",
                    f"Page Spanned: {page_span}",
                    f"Content: {content}",
                    "---",
                ])
            documents_text = "\n".join(docs_text_lines)
        else:
            documents_text = "(no relevant documents)"

        return self.answer_template.format(
            question=question,
            history=history_text,
            documents=documents_text,
            context_summary=context_summary or "(none)",
        )

    # ======== SUMMARY HELPERS ========
    def _summarize_history(self, messages: List[ChatMessage], previous_summary: str) -> str:
        """Summarize recent messages into a compact rolling context string."""
        recent_text = "\n".join([
            f"{'Human' if m.message_type == MessageType.QUESTION else 'Assistant'}: {m.content}"
            for m in messages[-8:]
        ]) if messages else "(no messages)"

        prompt = self.summary_template.format(
            previous_summary=previous_summary or "(none)",
            recent_messages=recent_text,
        )
        resp = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=180,
        )
        return (resp.choices[0].message.content or "").strip()

    def _update_session_context_summary(self, session_id: str, summary: str) -> None:
        """Persist the rolling summary to the session record."""
        try:
            self.table.update_item(
                Key={'PK': f"SESSION#{session_id}", 'SK': 'METADATA'},
                UpdateExpression='SET context_summary = :s, last_active = :ts',
                ExpressionAttributeValues={
                    ':s': summary,
                    ':ts': datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception as _:
            pass

    # ==================== MESSAGE MANAGEMENT ====================

    def get_session_messages(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages from a session"""
        try:
            response = self.table.query(
                KeyConditionExpression='PK = :pk AND begins_with(SK, :sk)',
                ExpressionAttributeValues={
                    ':pk': f"SESSION#{session_id}",
                    ':sk': 'MESSAGE#'
                },
                ScanIndexForward=False,
                Limit=limit
            )

            messages = []
            for item in reversed(response['Items']):
                messages.append(self._item_to_message(item))

            return messages
        except Exception as e:
            print(f"❌ Error getting session messages: {e}")
            return []

    def display_session_history(self, session_id: str, limit: int = 20):
        """Display recent conversation history for a session"""
        history = self.get_session_messages(session_id, limit)

        if not history:
            print("📭 No conversation history found")
            return

        print(f"\n📚 Session History ({len(history)} messages):")
        print("=" * 80)

        for msg in history:
            timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            role = "🧑 Human" if msg.message_type == MessageType.QUESTION else "🤖 Assistant"

            print(f"\n[{timestamp}] {role}:")

            if msg.message_type == MessageType.QUESTION and msg.refined_query and msg.refined_query != msg.original_content:
                print(f"Original: {msg.original_content}")
                print(f"Refined Query: {msg.refined_query}")
            else:
                content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                print(content)

    def clear_session_history(self, session_id: str) -> bool:
        """Clear all messages from a session"""
        try:
            messages = self.get_session_messages(session_id)

            with self.table.batch_writer() as batch:
                for message in messages:
                    batch.delete_item(
                        Key={
                            'PK': f"SESSION#{session_id}",
                            'SK': f"MESSAGE#{message.timestamp.isoformat()}#{message.message_id}"
                        }
                    )

            print(f"🗑️ Cleared {len(messages)} messages from session {session_id}")
            return True

        except Exception as e:
            print(f"❌ Error clearing session history: {e}")
            return False

    # ==================== HELPER METHODS ====================


    def _create_message(self, session_id: str, chatroom_id: str, user_id: str, message_type: MessageType,
                        content: str, original_content: str = None, refined_query: str = None,
                        retrieved_docs_count: int = 0) -> ChatMessage:
        """Create a new message"""
        return ChatMessage(
            message_id=str(uuid.uuid4()),
            session_id=session_id,
            chatroom_id=chatroom_id,
            user_id=user_id,
            message_type=message_type,
            original_content=original_content,
            refined_query=refined_query,
            content=content,
            timestamp=datetime.now(timezone.utc),
            retrieved_docs_count=retrieved_docs_count,
            metadata={}
        )

    def _update_session_status(self, session_id: str, status: SessionStatus) -> bool:
        """Update session status"""
        try:
            self.table.update_item(
                Key={'PK': f"SESSION#{session_id}", 'SK': 'METADATA'},
                UpdateExpression='SET #status = :status, last_active = :timestamp',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': status.value,
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
            return True
        except Exception as e:
            print(f"❌ Error updating session status: {e}")
            return False

    def _update_session_activity(self, session_id: str):
        """Update session last activity and message count"""
        try:
            self.table.update_item(
                Key={'PK': f"SESSION#{session_id}", 'SK': 'METADATA'},
                UpdateExpression='SET last_active = :timestamp, message_count = message_count + :inc',
                ExpressionAttributeValues={
                    ':timestamp': datetime.now(timezone.utc).isoformat(),
                    ':inc': 1
                }
            )
        except Exception as e:
            print(f"⚠️ Could not update session activity: {e}")

    def _increment_user_chatrooms(self, user_id: str):
        """Increment user's chatroom count"""
        try:
            self.table.update_item(
                Key={'PK': f"USER#{user_id}", 'SK': 'PROFILE'},
                UpdateExpression='SET total_chatrooms = total_chatrooms + :inc, last_active = :timestamp',
                ExpressionAttributeValues={
                    ':inc': 1,
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Could not update user chatroom count: {e}")

    def _increment_chatroom_sessions(self, chatroom_id: str):
        """Increment chatroom's session count"""
        try:
            self.table.update_item(
                Key={'PK': f"CHATROOM#{chatroom_id}", 'SK': 'METADATA'},
                UpdateExpression='SET total_sessions = total_sessions + :inc, active_sessions = active_sessions + :inc, last_active = :timestamp',
                ExpressionAttributeValues={
                    ':inc': 1,
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            print(f"⚠️ Could not update chatroom session count: {e}")

    # ==================== DYNAMODB CONVERSION METHODS ====================
    def _save_user(self, user: User):
        """Save user to DynamoDB"""
        self.table.put_item(Item={
            'PK': f"USER#{user.user_id}",
            'SK': 'PROFILE',
            'user_id': user.user_id,
            'username': user.user_name,
            'email': user.email,
            'created_at': user.created_at.isoformat(),
            'last_active': user.last_active.isoformat(),
            'total_chatrooms': user.total_chatrooms,
            'metadata': user.metadata or {}
        })

    def _save_chatroom(self, chatroom: Chatroom):
        """Save chatroom to DynamoDB"""
        self.table.put_item(Item={
            'PK': f"CHATROOM#{chatroom.chatroom_id}",
            'SK': 'METADATA',
            'chatroom_id': chatroom.chatroom_id,
            'user_id': chatroom.user_id,
            'chatroom_name': chatroom.chatroom_name,
            'description': chatroom.description,
            'created_at': chatroom.created_at.isoformat(),
            'last_active': chatroom.last_active.isoformat(),
            'total_sessions': chatroom.total_sessions,
            'active_sessions': chatroom.active_sessions,
            'metadata': chatroom.metadata or {}
        })

    def _save_session(self, session: Session):
        """Save session to DynamoDB"""
        item = {
            'PK': f"SESSION#{session.session_id}",
            'SK': 'METADATA',
            'session_id': session.session_id,
            'chatroom_id': session.chatroom_id,
            'user_id': session.user_id,
            'session_name': session.session_name,
            'status': session.status.value,
            'created_at': session.created_at.isoformat(),
            'last_active': session.last_active.isoformat(),
            'message_count': session.message_count,
            'context_summary': session.context_summary,
            'metadata': session.metadata or {}
        }

        if session.expires_at:
            item['expires_at'] = session.expires_at.isoformat()

        self.table.put_item(Item=item)

    def _save_message(self, message: ChatMessage):
        """Save message to DynamoDB"""
        item = {
            'PK': f"SESSION#{message.session_id}",
            'SK': f"MESSAGE#{message.timestamp.isoformat()}#{message.message_id}",
            'message_id': message.message_id,
            'session_id': message.session_id,
            'chatroom_id': message.chatroom_id,
            'user_id': message.user_id,
            'message_type': message.message_type.value,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'retrieved_docs_count': message.retrieved_docs_count,
            'metadata': message.metadata or {}
        }

        if message.original_content:
            item['original_content'] = message.original_content
        if message.refined_query:
            item['refined_query'] = message.refined_query

        self.table.put_item(Item=item)

    def _item_to_user(self, item: Dict) -> User:
        """Convert DynamoDB item to User"""
        return User(
            user_id=item['user_id'],
            username=item['username'],
            email=item['email'],
            created_at=datetime.fromisoformat(item['created_at']),
            last_active=datetime.fromisoformat(item['last_active']),
            total_chatrooms=item.get('total_chatrooms', 0),
            metadata=item.get('metadata', {})
        )

    def _item_to_chatroom(self, item: Dict) -> Chatroom:
        """Convert DynamoDB item to Chatroom"""
        return Chatroom(
            chatroom_id=item['chatroom_id'],
            user_id=item['user_id'],
            chatroom_name=item['chatroom_name'],
            description=item.get('description', ''),
            created_at=datetime.fromisoformat(item['created_at']),
            last_active=datetime.fromisoformat(item['last_active']),
            total_sessions=item.get('total_sessions', 0),
            active_sessions=item.get('active_sessions', 0),
            metadata=item.get('metadata', {})
        )

    def _item_to_session(self, item: Dict) -> Session:
        """Convert DynamoDB item to Session"""
        expires_at = None
        if 'expires_at' in item:
            expires_at = datetime.fromisoformat(item['expires_at'])

        return Session(
            session_id=item['session_id'],
            chatroom_id=item['chatroom_id'],
            user_id=item['user_id'],
            session_name=item['session_name'],
            status=SessionStatus(item['status']),
            created_at=datetime.fromisoformat(item['created_at']),
            last_active=datetime.fromisoformat(item['last_active']),
            expires_at=expires_at,
            message_count=item.get('message_count', 0),
            context_summary=item.get('context_summary', ''),
            metadata=item.get('metadata', {})
        )

    def _item_to_message(self, item: Dict) -> ChatMessage:
        """Convert DynamoDB item to ChatMessage"""
        return ChatMessage(
            message_id=item['message_id'],
            session_id=item['session_id'],
            chatroom_id=item['chatroom_id'],
            user_id=item['user_id'],
            message_type=MessageType(item['message_type']),
            original_content=item.get('original_content'),
            refined_query=item.get('refined_query'),
            content=item['content'],
            timestamp=datetime.fromisoformat(item['timestamp']),
            retrieved_docs_count=item.get('retrieved_docs_count', 0),
            metadata=item.get('metadata', {})
        )

    # ==================== UTILITY & ADMIN METHODS ====================

    def cleanup_expired_sessions(self):
        """Clean up expired sessions (background task)"""
        try:
            cutoff_time = datetime.now(timezone.utc)
            expired_count = 0

            # This is a simplified cleanup - in production you'd use a more efficient approach
            response = self.table.scan(
                FilterExpression='begins_with(SK, :sk)',
                ExpressionAttributeValues={':sk': 'METADATA'}
            )

            for item in response['Items']:
                if 'session_id' in item and 'expires_at' in item:
                    expires_at = datetime.fromisoformat(item['expires_at'])
                    if expires_at < cutoff_time and item.get('status') in ['active', 'paused']:
                        session_id = item['session_id']
                        self._update_session_status(session_id, SessionStatus.EXPIRED)
                        expired_count += 1

            print(f"🧹 Cleaned up {expired_count} expired sessions")
            return expired_count

        except Exception as e:
            print(f"❌ Error cleaning up sessions: {e}")
            return 0

    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            stats = {
                'total_users': 0,
                'total_chatrooms': 0,
                'total_sessions': 0,
                'active_sessions': 0,
            }

            response = self.table.scan()

            for item in response['Items']:
                if item['SK'] == 'PROFILE':
                    stats['total_users'] += 1
                elif item['SK'] == 'METADATA':
                    if 'chatroom_id' in item:
                        stats['total_chatrooms'] += 1
                    elif 'session_id' in item:
                        stats['total_sessions'] += 1
                        if item.get('status') == 'active':
                            stats['active_sessions'] += 1

            return stats

        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {}

    def display_user_summary(self, user_id: str):
        """Display a summary for a user"""
        user = self.get_user(user_id)
        if not user:
            print(f"❌ User {user_id} not found")
            return

        chatrooms = self.list_user_chatrooms(user_id)

        print(f"\n👤 User Summary: {user.user_name}")
        print("=" * 50)
        print(f"Email: {user.email}")
        print(f"Created: {user.created_at.strftime('%Y-%m-%d %H:%M')}")
        print(f"Last Active: {user.last_active.strftime('%Y-%m-%d %H:%M')}")
        print(f"Total Chatrooms: {user.total_chatrooms}")

        if chatrooms:
            print(f"\n🏠 Chatrooms:")
            for chatroom in chatrooms[:10]:
                sessions = self.list_chatroom_sessions(chatroom.chatroom_id)
                active_count = len([s for s in sessions if s.status == SessionStatus.ACTIVE])
                print(f"  • {chatroom.chatroom_name}: {len(sessions)} sessions ({active_count} active)")

        print("=" * 50)

    # ==================== CONVENIENCE WRAPPER ====================

class SimpleRAGChatWrapper:
    """
    Simplified wrapper around the enhanced system for single-user, single-chatroom usage
    This provides backward compatibility with your existing SimpleChat interface
    """

    def __init__(self, retriever, groq_api_key: str, **kwargs):
        self.enhanced_chat = EnhancedRAGChat(retriever, groq_api_key, **kwargs)

        # Create default user and chatroom
        self.user_id = self.enhanced_chat.create_user("default_user", "user@example.com")
        self.chatroom_id = self.enhanced_chat.create_chatroom(
            self.user_id, "Main Chatroom", "Default cybersecurity chat"
        )
        self.session_id = self.enhanced_chat.create_session(
            self.chatroom_id, self.user_id, "Main Session"
        )

        print(f"✅ Simple wrapper initialized with session: {self.session_id}")

    def chat(self, question: str, retrieval_k: int = 5) -> str:
        """Simple chat method - compatible with your existing code"""
        return self.enhanced_chat.chat(self.session_id, self.user_id, question, retrieval_k)

    def display_history(self, limit: int = 20):
        """Display conversation history"""
        return self.enhanced_chat.display_session_history(self.session_id, limit)

    def clear_history(self) -> bool:
        """Clear conversation history"""
        return self.enhanced_chat.clear_session_history(self.session_id)

    def get_history(self, limit: int = 10) -> List[ChatMessage]:
        """Get conversation history"""
        return self.enhanced_chat.get_session_messages(self.session_id, limit)






def main():
    """Main function to run the enhanced chat system"""
    print("🚀 Initializing Enhanced RAG Chat System...")
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

        # Choose between full system or simple wrapper
        use_simple_wrapper = input("\nUse simple wrapper? (y/n): ").strip().lower() == 'y'

        if use_simple_wrapper:
            # Simple wrapper for backward compatibility
            chat = SimpleRAGChatWrapper(
                retriever=rag_retriever,
                groq_api_key=os.getenv('GROQ_API_KEY'),
                dynamodb_table_name='enhanced_cybersecurity_chat'
            )

            print("\n" + "=" * 60)
            print("✅ Simple chat system ready!")
            print("Commands: 'history', 'clear', 'quit'")
            print("=" * 60)

            # Simple chat loop
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

                    response = chat.chat(user_input)
                    print(f"\n🤖 Assistant: {response}")

                except KeyboardInterrupt:
                    print("\n👋 Chat interrupted. Goodbye!")
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")

        else:
            # Full enhanced system with multi-user support
            chat_system = EnhancedRAGChat(
                retriever=rag_retriever,
                groq_api_key=os.getenv('GROQ_API_KEY'),
                dynamodb_table_name='enhanced_cybersecurity_chat'
            )

            print("\n" + "=" * 60)
            print("✅ Enhanced system ready! Starting interactive demo...")
            print("Commands: create-user, create-chatroom, create-session, chat, list, stats, quit")
            print("=" * 60)

            demo_workflow(chat_system)

    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("Make sure:")
        print("1. DynamoDB Local is running: docker run -p 8000:8000 amazon/dynamodb-local")
        print("2. FAISS index exists in 'faiss_index' directory")
        print("3. Environment variables are set in .env file")


def demo_workflow(chat_system: EnhancedRAGChat):
    """Interactive demo workflow for the enhanced system"""
    current_user_id = None
    current_chatroom_id = None
    current_session_id = None

    while True:
        try:
            cmd = input(f"\n📝 Command [{current_user_id or 'no-user'}]: ").strip().lower()

            if cmd == 'quit':
                print("👋 Goodbye!")
                break

            elif cmd == 'create-user':
                username = input("Username: ").strip()
                email = input("Email: ").strip()
                current_user_id = chat_system.create_user(username, email)
                print(f"✅ Created user: {current_user_id}")

            elif cmd == 'create-chatroom':
                if not current_user_id:
                    print("❌ Please create a user first")
                    continue
                name = input("Chatroom name: ").strip()
                desc = input("Description (optional): ").strip()
                current_chatroom_id = chat_system.create_chatroom(current_user_id, name, desc)
                print(f"✅ Created chatroom: {current_chatroom_id}")

            elif cmd == 'create-session':
                if not current_chatroom_id:
                    print("❌ Please create a chatroom first")
                    continue
                name = input("Session name: ").strip()
                current_session_id = chat_system.create_session(current_chatroom_id, current_user_id, name)
                print(f"✅ Created session: {current_session_id}")

            elif cmd == 'chat':
                if not current_session_id:
                    print("❌ Please create a session first")
                    continue
                question = input("Your question: ").strip()
                if question:
                    answer = chat_system.chat(current_session_id, current_user_id, question)
                    print(f"\n🤖 Assistant: {answer}")

            elif cmd == 'list':
                if current_user_id:
                    chat_system.display_user_summary(current_user_id)
                else:
                    print("❌ No current user")

            elif cmd == 'history':
                if current_session_id:
                    chat_system.display_session_history(current_session_id, limit=10)
                else:
                    print("❌ No current session")

            elif cmd == 'stats':
                stats = chat_system.get_system_stats()
                print("\n📊 System Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")

            elif cmd == 'help':
                print("\nAvailable commands:")
                print("  create-user     - Create a new user")
                print("  create-chatroom - Create a chatroom for current user")
                print("  create-session  - Create a session in current chatroom")
                print("  chat            - Send message to current session")
                print("  list            - List current user's chatrooms/sessions")
                print("  history         - Show current session history")
                print("  stats           - Show system statistics")
                print("  quit            - Exit")

            else:
                print("❌ Unknown command. Type 'help' for available commands.")

        except KeyboardInterrupt:
            print("\n👋 Demo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()