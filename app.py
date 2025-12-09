"""
Web interface for Python Q&A Chatbot using Streamlit
Run with: streamlit run app.py
"""

import streamlit as st
from word2vec_chatbot import Word2VecQAChatbot
from question_validator import QuestionValidator
import os


# Page configuration
st.set_page_config(
    page_title="Python Q&A Chatbot",
    page_icon="🐍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .answer-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_word2vec_chatbot():
    """Load Word2Vec chatbot model (cached for performance)"""
    chatbot = Word2VecQAChatbot()
    
    if os.path.exists(chatbot.model_path) and os.path.exists(chatbot.embeddings_path):
        if chatbot.load_model():
            return chatbot, "Word2Vec model loaded successfully!"
    
    return None, "Word2Vec model files not found"


@st.cache_resource
def load_question_validator():
    """Load question validator (cached)"""
    return QuestionValidator()


def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.6:
        return "confidence-high"
    elif confidence >= 0.3:
        return "confidence-medium"
    else:
        return "confidence-low"


def main():
    # Header
    st.markdown('<h1 class="main-header">🐍 Python Q&A Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Custom Word2Vec • LLM-Free • Smart Input Validation</p>', unsafe_allow_html=True)
    
    # Load question validator
    validator = load_question_validator()
    
    # Load Word2Vec chatbot
    with st.spinner("Loading Custom Word2Vec model..."):
        chatbot, status = load_word2vec_chatbot()
        model_name = "Custom Word2Vec"
        model_desc = "Trained from scratch on YOUR data"
    
    if chatbot is None:
        st.error(status)
        st.info("💡 Word2Vec model not found. Make sure you've placed 'word2vec_custom.model' and 'word2vec_chatbot_model.pkl' in the project folder.")
        st.stop()
    
    # Sidebar - About and Stats
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This chatbot answers Python programming questions without using any LLMs.
        
        **Features:**
        - Custom Word2Vec (trained from scratch)
        - Semantic understanding
        - Cosine similarity matching
        - Smart input validation
        """)
        
        st.header("📊 Stats")
        stats = chatbot.get_stats() if hasattr(chatbot, 'get_stats') else {}
        
        if chatbot.df is not None:
            st.metric("Total Q&A Pairs", f"{len(chatbot.df):,}")
        
        st.metric("Vocabulary Size", f"{stats.get('vocabulary_size', 0):,}")
        st.metric("Vector Dimensions", stats.get('vector_size', 0))
        st.metric("Model Type", model_name)
        st.metric("Pre-trained?", "❌ NO (Trained by YOU)")
        
        st.header("💡 Tips")
        st.write("""
        - **Start with question words** (How, What, Why)
        - **Include Python-related terms** for better results
        - **Be specific** with your questions
        - **Use complete sentences** for validation
        - System will detect invalid/random inputs
        """)
        
        st.header("✅ Valid Input Examples")
        st.write("""
        **Questions:**
        ✓ How do I sort a list?
        ✓ What is a lambda function?
        ✓ Explain list comprehension
        
        **Casual Chat:**
        ✓ Hi / Hello / Hey
        ✓ Thanks / Thank you
        ✓ Help / Bye
        
        **Invalid:**
        ✗ python (too short)
        ✗ asdfgh (gibberish)
        ✗ 12345 (just numbers)
        """)
        
        st.header("🔧 Settings")
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)
        top_k = st.slider("Number of Matches to Consider", 1, 10, 5)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("is_error", False):
                # This was a validation error - already formatted
                pass
            elif message.get("is_greeting", False) or message.get("is_casual", False):
                # This was a greeting or casual conversation - no confidence needed
                pass
            elif "confidence" in message:
                confidence_class = get_confidence_class(message["confidence"])
                st.markdown(f'<p class="{confidence_class}">Confidence: {message["confidence"]:.2%}</p>', 
                          unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about Python..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Validate the question first
        is_valid, reason, validation_confidence = validator.is_valid_question(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            if not is_valid:
                # Invalid question - show helpful message
                st.warning(f"⚠️ {reason}")
                st.info(validator.get_suggestion(prompt))
                
                # Add validation error to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ **Input Validation Failed**\n\n{reason}\n\n{validator.get_suggestion(prompt)}",
                    "is_error": True
                })
            else:
                # Check if it's a greeting or casual conversation
                if reason == "greeting_detected":
                    # Handle greeting with a friendly response
                    greeting_responses = [
                        "👋 Hello! I'm your Python Q&A assistant. How can I help you learn Python today?",
                        "Hi there! 😊 Ready to answer your Python questions!",
                        "Hello! 🐍 Ask me anything about Python programming!",
                        "Hey! 👋 What Python topic would you like to explore?",
                    ]
                    import random
                    response = random.choice(greeting_responses)
                    st.markdown(response)
                    
                    # Add greeting response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "is_greeting": True
                    })
                elif reason == "casual_conversation":
                    # Handle casual phrases
                    casual_responses = {
                        'thanks': "You're welcome! 😊 Feel free to ask more Python questions!",
                        'thank you': "You're welcome! Happy to help with Python! 🐍",
                        'bye': "Goodbye! 👋 Come back anytime you need Python help!",
                        'goodbye': "See you later! 👋 Happy coding!",
                        'help': "I'm here to answer Python programming questions! Try asking about functions, loops, data types, or any Python concept.",
                        'ok': "Great! What would you like to know about Python?",
                        'okay': "Alright! Ask me anything about Python programming!",
                        'yes': "Awesome! What's your Python question?",
                        'cool': "Thanks! 😎 What Python topic interests you?",
                        'nice': "Thank you! Ask me anything about Python!",
                    }
                    
                    # Find matching casual phrase
                    prompt_lower = prompt.lower()
                    response = None
                    for key, value in casual_responses.items():
                        if key in prompt_lower:
                            response = value
                            break
                    
                    if not response:
                        response = "I'm here to help with Python! What would you like to know?"
                    
                    st.markdown(response)
                    
                    # Add casual response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "is_casual": True
                    })
                else:
                    # Valid question - get answer using Word2Vec
                    with st.spinner("Thinking..."):
                        answer, confidence = chatbot.get_answer(prompt)
                    
                    st.markdown(answer)
                    confidence_class = get_confidence_class(confidence)
                    st.markdown(f'<p class="{confidence_class}">Confidence: {confidence:.2%}</p>', 
                               unsafe_allow_html=True)
                    
                    # Show validation info if borderline
                    if validation_confidence < 0.8:
                        st.caption(f"💡 {reason}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "confidence": confidence
                    })
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()
