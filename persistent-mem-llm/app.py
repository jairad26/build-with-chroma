from flask import Flask, render_template, request, jsonify
from persistent_llm import ConversationMemoryGraph
import os
from dotenv import load_dotenv
from flask_cors import CORS
from collections import OrderedDict
import json

load_dotenv()

app = Flask(__name__)
CORS(app)
memory_system = ConversationMemoryGraph()

@app.route('/')
def home():
    # Get all unique conversations from ChromaDB
    all_conversations = memory_system.collection.get()
    
    # Group by conversation_id and get latest message for each
    conversation_list = {}
    for doc, metadata in zip(all_conversations['documents'], all_conversations['metadatas']):
        conv_id = metadata['conversation_id']
        timestamp = metadata['timestamp']
        if conv_id not in conversation_list or timestamp > conversation_list[conv_id]['timestamp']:
            conversation_list[conv_id] = {
                'last_message': doc[:100] + '...' if len(doc) > 100 else doc,
                'timestamp': timestamp
            }
        sorted_conversations = OrderedDict(
        sorted(conversation_list.items(), 
              key=lambda x: x[1]['timestamp'],
              reverse=True)
    )
    
    return render_template('chat.html', conversations=conversation_list)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message')
    conversation_id = data.get('conversation_id')
    
    result = memory_system.process_message(message, conversation_id)
    return jsonify(result)

@app.route('/conversation/<conversation_id>')
def get_conversation(conversation_id):
    # Get all messages for a specific conversation
    messages = memory_system.collection.get(
        where={"conversation_id": conversation_id}
    )
    
    # Format messages
    conversation = []
    similar_conversations = []
    
    for doc, metadata in zip(messages['documents'], messages['metadatas']):
        role = metadata.get("role", "human")
        
        conversation.append({
            'content': doc,
            'role': role,
            'timestamp': metadata['timestamp']
        })
        
        # Get similar conversations from metadata if they exist
        if metadata.get("similar_conversations"):
            try:
                similar_ids = json.loads(metadata["similar_conversations"])
                similarity_scores = json.loads(metadata.get("similarity_scores", "[1.0]"))
                
                for sim_id, score in zip(similar_ids, similarity_scores):
                    if sim_id not in [s["id"] for s in similar_conversations]:
                        # Get the first message of the similar conversation
                        sim_messages = memory_system.collection.get(
                            where={"conversation_id": sim_id},
                            limit=1
                        )
                        if sim_messages["documents"]:
                            similar_conversations.append({
                                "id": sim_id,
                                "preview": sim_messages["documents"][0][:100] + "...",
                                "similarity_score": score,
                                "timestamp": sim_messages["metadatas"][0]["timestamp"]
                            })
            except json.JSONDecodeError:
                # Handle case where the JSON string is invalid
                print(f"Error decoding similar conversations for conversation {conversation_id}")
    
    # Sort by timestamp
    conversation.sort(key=lambda x: x['timestamp'])
    
    return jsonify({
        "messages": conversation,
        "similar_conversations": similar_conversations
    })

if __name__ == '__main__':
    app.run(debug=True) 