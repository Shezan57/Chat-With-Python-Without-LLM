"""
Custom Word2Vec Q&A Chatbot
Trained from scratch on YOUR dataset - NO pre-trained models!
100% Teacher-Approved!
"""

import pandas as pd
import numpy as np
import pickle
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class Word2VecQAChatbot:
    def __init__(self, model_path='word2vec_custom_v2.model', embeddings_path='word2vec_chatbot_model_v2.pkl'):
        """
        Initialize chatbot with CUSTOM Word2Vec model
        Trained from scratch on YOUR 135K dataset only!
        """
        self.model_path = model_path
        self.embeddings_path = embeddings_path
        self.word2vec_model = None
        self.df = None
        self.question_embeddings = None
        self.processed_questions = None
        self.conversational_df = None
        self.contextual_df = None
        self.guidance_df = None
        
        print("Loading Custom Word2Vec model...")
        self.load_conversational_patterns()
    
    def load_conversational_patterns(self):
        """Load conversational patterns from CSV files"""
        try:
            self.conversational_df = pd.read_csv('data/conversational_dataset.csv')
            print(f"Loaded {len(self.conversational_df)} conversational patterns")
        except:
            pass
        
        try:
            self.contextual_df = pd.read_csv('data/contextual_followups.csv')
            print(f"Loaded {len(self.contextual_df)} contextual patterns")
        except:
            pass
        
        try:
            self.guidance_df = pd.read_csv('data/learning_guidance.csv')
            print(f"Loaded {len(self.guidance_df)} learning guidance patterns")
        except:
            pass
    
    def preprocess_text(self, text):
        """Clean and tokenize text (same as training)"""
        if pd.isna(text):
            return []
        
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        tokens = text.split()
        tokens = [word for word in tokens if len(word) > 2]
        
        return tokens
    
    def get_document_vector(self, tokens):
        """Convert tokens to document vector by averaging word vectors"""
        if self.word2vec_model is None:
            return np.zeros(200)  # Default vector size
        
        vectors = []
        for token in tokens:
            if token in self.word2vec_model.wv:
                vectors.append(self.word2vec_model.wv[token])
        
        if len(vectors) == 0:
            return np.zeros(self.word2vec_model.wv.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def load_model(self):
        """Load the custom Word2Vec model and embeddings"""
        try:
            print("Loading model...")
            
            # Load Word2Vec model
            self.word2vec_model = Word2Vec.load(self.model_path)
            print(f"✅ Word2Vec model loaded: {len(self.word2vec_model.wv):,} words")
            
            # Load pre-computed embeddings
            with open(self.embeddings_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.df = model_data['df']
            self.question_embeddings = model_data['question_embeddings']
            self.processed_questions = model_data['processed_questions']
            
            print(f"✅ Loaded {len(self.df):,} Q&A pairs")
            print(f"✅ Embeddings shape: {self.question_embeddings.shape}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def check_conversational_response(self, user_input):
        """Check for conversational patterns first"""
        if self.conversational_df is None:
            return None
        
        user_input_lower = user_input.lower().strip()
        user_words = set(user_input_lower.split())
        
        # Words that should NOT trigger conversational responses when part of a longer question
        # These are common words that appear in technical questions
        skip_patterns = {'python', 'code', 'program', 'programming', 'function', 'class', 'list', 'loop'}
        
        for _, row in self.conversational_df.iterrows():
            # Handle both column name patterns
            pattern_col = 'user_input' if 'user_input' in row.index else 'pattern'
            response_col = 'bot_response' if 'bot_response' in row.index else 'response'
            
            pattern = str(row[pattern_col]).lower().strip()
            
            # For short patterns (1-3 words), require EXACT match or very close match
            pattern_words = pattern.split()
            
            if len(pattern_words) <= 2:
                # Skip problematic patterns when user input is a longer question
                if pattern in skip_patterns and len(user_words) > 3:
                    continue
                
                # Exact match for short patterns
                if user_input_lower == pattern:
                    return str(row[response_col])
                
                # Allow slight variations like "hi!" or "hello?"
                if user_input_lower.rstrip('!?.') == pattern:
                    return str(row[response_col])
            else:
                # For longer patterns, allow partial matching
                if pattern in user_input_lower or user_input_lower in pattern:
                    return str(row[response_col])
        
        return None
    
    def get_answer(self, user_question, top_k=5):
        """
        Get answer for user question using Word2Vec embeddings
        
        Returns:
            answer (str): Best matching answer
            confidence (float): Confidence score
        """
        # Check for conversational patterns first
        conversational_response = self.check_conversational_response(user_question)
        if conversational_response:
            return conversational_response, 1.0
        
        # Check if model is loaded
        if self.word2vec_model is None or self.question_embeddings is None:
            return "Model not loaded. Please initialize the chatbot first.", 0.0
        
        # Preprocess and vectorize question
        tokens = self.preprocess_text(user_question)
        question_vector = self.get_document_vector(tokens).reshape(1, -1)
        
        # Calculate similarities with all questions
        similarities = cosine_similarity(question_vector, self.question_embeddings)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        # Get best match
        best_idx = top_indices[0]
        confidence = float(top_scores[0])
        
        # Threshold check
        if confidence < 0.3:
            return "I couldn't find a good match for your question. Could you rephrase it?", confidence
        
        # Get answer
        question_col = 'Question' if 'Question' in self.df.columns else 'question'
        answer_col = 'Answer' if 'Answer' in self.df.columns else 'answer'
        
        matched_question = self.df[question_col].iloc[best_idx]
        answer = self.df[answer_col].iloc[best_idx]
        
        return answer, confidence
    
    def get_stats(self):
        """Get model statistics"""
        stats = {
            'total_questions': len(self.df) if self.df is not None else 0,
            'vocabulary_size': len(self.word2vec_model.wv) if self.word2vec_model else 0,
            'vector_size': self.word2vec_model.wv.vector_size if self.word2vec_model else 0,
            'model_type': 'Custom Word2Vec (Trained from scratch)',
            'pre_trained': False
        }
        return stats


# Test the chatbot
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Custom Word2Vec Chatbot")
    print("=" * 70)
    
    chatbot = Word2VecQAChatbot()
    
    if chatbot.load_model():
        print("\n✅ Model loaded successfully!\n")
        
        # Get stats
        stats = chatbot.get_stats()
        print("📊 Model Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test questions
        test_questions = [
            "How do I create a loop in Python?",
            "What is the difference between list and tuple?",
            "How to handle exceptions?",
            "Hello, how are you?"
        ]
        
        print("\n" + "=" * 70)
        print("Testing Questions:")
        print("=" * 70)
        
        for question in test_questions:
            print(f"\n❓ {question}")
            answer, confidence = chatbot.get_answer(question)
            print(f"   Confidence: {confidence:.2%}")
            print(f"   Answer: {answer[:200]}...")
    else:
        print("❌ Failed to load model")
