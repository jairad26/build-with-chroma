from typing import TypedDict, Sequence, List, Dict, Optional
from datetime import datetime
from uuid import uuid4
import chromadb
from chromadb.config import Settings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, StateGraph
from enum import Enum
import os
from dotenv import load_dotenv
import json  # Add this import at the top

load_dotenv()

# Define state for the conversation memory system
class ConversationState(TypedDict):
    messages: Sequence[BaseMessage]
    current_input: str
    conversation_id: str
    similar_conversations: List[Dict]
    summaries: List[str]
    final_response: Optional[str]
    metadata: Optional[Dict]
    should_use_context: bool

# Define possible actions
class Action(str, Enum):
    FIND_SIMILAR = "find_similar"
    SUMMARIZE = "summarize"
    GENERATE = "generate"
    STORE = "store"
    END = "end"

class ConversationMemoryGraph:
    def __init__(
        self,
        persist_directory: str = "./chroma_conversations",
        collection_name: str = "conversation_history",
        similarity_threshold: float = 0.6,
        openai_api_key: str = os.getenv("OPENAI_API_KEY")
    ):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            temperature=0,
            model="claude-3-5-sonnet-20240620",
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        self.similarity_threshold = similarity_threshold
        self.workflow = self._create_workflow()

    def _find_similar_conversations(self, state: ConversationState) -> ConversationState:
        """Node: Find similar previous conversations"""
        print("\n=== Starting Similarity Search ===")
        print(f"Searching for messages similar to: {state['current_input']}")
        
        # First, find similar questions (human messages)
        results = self.collection.query(
            query_texts=[state["current_input"]],
            n_results=3,
            where={"role": "human"}  # Only search through human messages
        )
        
        print(f"\nFound {len(results['documents'][0])} potential matches")
        
        similar_conversations = []
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            similarity_score = 1 - distance
            print(f"\nPotential match:")
            print(f"- Message: {doc[:100]}...")
            print(f"- Similarity score: {similarity_score:.3f}")
            print(f"- Threshold: {self.similarity_threshold}")
            
            if similarity_score >= self.similarity_threshold:
                print("✓ Match accepted (above threshold)")
                # Get the conversation_id from metadata
                conv_id = metadata["conversation_id"]
                print(f"- Conversation ID: {conv_id}")
                
                # Fetch all messages from this conversation
                conv_messages = self.collection.get(
                    where={"conversation_id": conv_id},
                )
                
                # Sort messages by timestamp and organize into conversation format
                messages = []
                for msg_content, msg_metadata in zip(conv_messages['documents'], conv_messages['metadatas']):
                    messages.append({
                        "role": msg_metadata["role"],
                        "content": msg_content,
                        "timestamp": msg_metadata["timestamp"]
                    })
                
                # Sort by timestamp to maintain conversation order
                messages.sort(key=lambda x: x["timestamp"])
                
                similar_conversations.append({
                    "conversation": messages,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                })
            else:
                print("✗ Match rejected (below threshold)")
        
        print(f"\nTotal similar conversations found: {len(similar_conversations)}")
        print("=== Similarity Search Complete ===\n")
        
        return {
            **state,
            "similar_conversations": similar_conversations
        }

    def _summarize_conversations(self, state: ConversationState) -> ConversationState:
        """Node: Generate summaries for similar conversations"""
        if not state["similar_conversations"]:
            return {**state, "summaries": []}
            
        template = """Provide a concise summary of this conversation, 
        focusing on key points and conclusions:

        {conversation}

        Summary:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        summaries = []
        for conv in state["similar_conversations"]:
            conversation_text = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in conv["conversation"]
            )
            
            summary = chain.invoke({"conversation": conversation_text})
            summaries.append({
                "summary": summary,
                "timestamp": conv["metadata"]["timestamp"],
                "similarity": conv["similarity_score"]
            })
        
        return {**state, "summaries": summaries}

    def _generate_response(self, state: ConversationState) -> ConversationState:
        """Node: Generate response using context if available"""
        messages = state.get("messages", [])
        
        if messages:
            conversation_history = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages
            )
            conversation_context = f"\nConversation history:\n{conversation_history}\n"
        else:
            conversation_context = ""
        
        if state["summaries"] and state["should_use_context"]:
            # This is a new conversation with similar past conversations
            template = """You have access to summaries of previous related conversations and the current conversation history.
            You should explicitly mention when you're using information from previous conversations.
            If there is no previous conversation, don't say anything about "no prior context", just answer the question, the user
            should not be made aware that there is no conversation history.

            {conversation_context}
            
            Previous relevant conversations:
            {summaries}

            Current query: {query}

            Please start your response by acknowledging the similar conversations you found, 
            and then provide your answer incorporating insights from both the previous conversations 
            and the current context. Use phrases like "Based on our previous conversation about..." 
            or "I see we've discussed something similar before..." to make it clear you're using 
            historical context.

            Response:"""
            
            summaries_text = "\n".join(
                f"\n---\nPrevious conversation ({s['timestamp']}, "
                f"similarity: {s['similarity']:.2f}):\n{s['summary']}"
                for s in state["summaries"]
            )
        else:
            # This is either an ongoing conversation or a new conversation without similar ones
            template = """Please provide a response to the following query, taking into account 
            any existing conversation context. if there is no conversation history, just answer the question, the user
            should not be made aware that there is no conversation history:
            
            {conversation_context}
            Query: {query}

            If there is conversation history, make sure to maintain continuity and reference 
            previous parts of the conversation when relevant.

            Response:"""
            summaries_text = ""
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        
        response = chain.invoke({
            "summaries": summaries_text,
            "query": state["current_input"],
            "conversation_context": conversation_context
        })
        
        return {**state, "final_response": response}

    def _store_conversation(self, state: ConversationState) -> ConversationState:
        """Node: Store the conversation in ChromaDB"""
        current_time = datetime.now().isoformat()
        
        # Create base metadata
        base_metadata = {
            "conversation_id": state["conversation_id"],
            "timestamp": current_time,
        }
        
        # Add similar conversation IDs to metadata if they exist
        if state["similar_conversations"]:
            similar_ids = [conv["metadata"]["conversation_id"] 
                          for conv in state["similar_conversations"]]
            # Convert lists to JSON strings
            base_metadata["similar_conversations"] = json.dumps(similar_ids)
            base_metadata["similarity_scores"] = json.dumps([
                conv["similarity_score"] for conv in state["similar_conversations"]
            ])
        
        # Store human message
        human_metadata = {
            **base_metadata,
            "role": "human",
            "message_type": "query",
        }
        
        # Store AI response
        ai_metadata = {
            **base_metadata,
            "role": "ai",
            "message_type": "response",
        }
        
        # Store each message separately with its metadata
        base_id = f"conv_{state['conversation_id']}"
        self.collection.add(
            documents=[state["current_input"], state["final_response"]],
            metadatas=[human_metadata, ai_metadata],
            ids=[f"{base_id}_human_{uuid4()}", f"{base_id}_ai_{uuid4()}"]
        )
        
        return state

    def _should_continue(self, state: ConversationState) -> Action:
        """Determine next action based on state"""
        if not state.get("similar_conversations"):
            return Action.GENERATE
        
        if not state.get("summaries"):
            return Action.SUMMARIZE
            
        if not state.get("final_response"):
            return Action.GENERATE
            
        if state.get("final_response"):
            return Action.STORE
            
        return Action.END

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("find_similar", self._find_similar_conversations)
        workflow.add_node("summarize", self._summarize_conversations)
        workflow.add_node("generate", self._generate_response)
        workflow.add_node("store", self._store_conversation)
        
        # Add edges
        workflow.add_edge("find_similar", "summarize")
        workflow.add_edge("summarize", "generate")
        workflow.add_edge("generate", "store")
        
        # Set entry point
        workflow.set_entry_point("find_similar")
        
        # Compile
        return workflow.compile()

    def process_message(self, message: str, conversation_id: Optional[str] = None) -> Dict:
        """Process a message and return the response with conversation context"""
        print("\n=== Processing New Message ===")
        print(f"Message: {message}")
        print(f"Conversation ID: {conversation_id or 'New conversation'}")
        
        # If conversation_id is provided, get previous messages from this conversation
        messages = []
        if conversation_id:
            previous_messages = self.collection.get(
                where={"conversation_id": conversation_id},
            )
            
            # Convert to list of messages and sort by timestamp
            for doc, metadata in zip(previous_messages['documents'], previous_messages['metadatas']):
                messages.append({
                    "content": doc,
                    "role": metadata["role"],
                    "timestamp": metadata["timestamp"]
                })
            messages.sort(key=lambda x: x["timestamp"])
            print(f"Found {len(messages)} previous messages in this conversation")
            print("Skipping similarity search for existing conversation")
        
        # Initialize conversation state
        state: ConversationState = {
            "current_input": message,
            "conversation_id": conversation_id or str(uuid4()),
            "similar_conversations": [],
            "summaries": [],
            "metadata": {
                "timestamp": datetime.now().isoformat(),
            },
            "final_response": None,
            "should_use_context": not bool(messages),  # Only use similar conversation context for new conversations
            "messages": messages  # Always include current conversation messages
        }
        
        if messages:
            # For existing conversations, skip the workflow and just generate response
            print("\nGenerating response with conversation history...")
            template = """Here is the conversation history:
            {conversation_history}
            
            Current query: {query}
            
            Please provide a response that takes into account the conversation history 
            and maintains continuity with the ongoing discussion. Make references to 
            previous parts of the conversation when relevant.
            
            Response:"""
            
            conversation_history = "\n".join(
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages
            )
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()
            
            response = chain.invoke({
                "conversation_history": conversation_history,
                "query": message
            })
            
            state["final_response"] = response
            
            # Store the conversation without running the full workflow
            self._store_conversation(state)
        else:
            # For new conversations, run the full workflow
            print("\nRunning workflow for new conversation...")
            state = self.workflow.invoke(state)
        
        # Return results including conversation_id
        return {
            "response": state["final_response"],
            "found_similar": len(state["similar_conversations"]) > 0,
            "similar_conversations": state["similar_conversations"],
            "conversation_id": state["conversation_id"]
        }

# Example usage
if __name__ == "__main__":
    load_dotenv()

    # Initialize system
    memory_system = ConversationMemoryGraph()
    
    # First conversation about Flask authentication
    result1 = memory_system.process_message(
        "What's the best way to implement authentication in a Flask app?",
        conversation_id="1"
    )
    print("Response 1:", result1["response"])
    
    # Later conversation about Flask authentication (similar topic)
    result2 = memory_system.process_message(
        "How do I handle user authentication in Flask?",
        conversation_id="1"
    )
        
    if result2["found_similar"]:
        print("\nFound similar previous conversations!")
        for conv in result2["similar_conversations"]:
            print(f"\nPrevious conversation from {conv['timestamp']}")
            print(f"Similarity score: {conv['similarity']:.2f}")
            print(f"Summary: {conv['summary']}")