"""
Question Validator Module
Analyzes user input to determine if it's a valid question or random text
"""

import re
from typing import Tuple


class QuestionValidator:
    """Validates if user input is a meaningful question"""
    
    def __init__(self):
        # Greetings and conversational phrases (always valid)
        self.greetings = {
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'howdy', 'hiya', 'sup', 'wassup', "what's up",
            'yo', 'hola', 'namaste'
        }
        
        # Casual conversation starters (valid interactions)
        self.casual_phrases = {
            'thanks', 'thank you', 'bye', 'goodbye', 'see you', 'later',
            'ok', 'okay', 'yes', 'no', 'sure', 'great', 'cool', 'nice',
            'help', 'help me', 'assist', 'assist me'
        }
        
        # Question words (WH-words and auxiliary verbs)
        self.question_words = {
            'what', 'when', 'where', 'who', 'whom', 'whose', 'which', 'why', 'how',
            'is', 'are', 'am', 'was', 'were', 'do', 'does', 'did', 'can', 'could',
            'will', 'would', 'should', 'shall', 'may', 'might', 'must',
            'has', 'have', 'had', 'tell', 'explain', 'describe', 'show', 'define'
        }
        
        # Programming-related keywords (Python-specific)
        self.programming_keywords = {
            'python', 'code', 'program', 'function', 'class', 'method', 'variable',
            'list', 'dict', 'dictionary', 'tuple', 'set', 'string', 'integer', 'float',
            'loop', 'for', 'while', 'if', 'else', 'elif', 'try', 'except', 'import',
            'def', 'return', 'print', 'input', 'file', 'read', 'write', 'open',
            'lambda', 'map', 'filter', 'reduce', 'comprehension', 'generator',
            'decorator', 'iterator', 'module', 'package', 'library', 'framework',
            'error', 'exception', 'debug', 'syntax', 'runtime', 'compile',
            'array', 'sort', 'search', 'algorithm', 'data', 'type', 'object',
            'inheritance', 'polymorphism', 'encapsulation', 'oop', 'api',
            'json', 'xml', 'database', 'sql', 'web', 'http', 'request',
            'pandas', 'numpy', 'matplotlib', 'django', 'flask', 'tensorflow'
        }
        
        # Command-like patterns (e.g., "sort list", "create function")
        self.command_patterns = [
            r'\b(sort|create|make|build|write|generate|implement|design|develop)\b.*\b(list|function|class|program|code|script|application)\b',
            r'\b(how to|how do i|how can i|how should i)\b',
            r'\b(help me|help with|assist with)\b',
        ]
        
        # Minimum meaningful length
        self.min_length = 3
        
    def is_valid_question(self, text: str) -> Tuple[bool, str, float]:
        """
        Validate if the input is a meaningful question
        
        Returns:
            Tuple of (is_valid, reason, confidence)
            - is_valid: Boolean indicating if it's a valid question
            - reason: String explanation of why it's valid/invalid
            - confidence: Float between 0-1 indicating confidence in validation
        """
        if not text or not text.strip():
            return False, "Empty input", 1.0
        
        # Clean and normalize text
        text_clean = text.strip().lower()
        words = re.findall(r'\b\w+\b', text_clean)
        
        # Check for greetings (always valid and friendly response)
        if text_clean in self.greetings or any(greeting in text_clean for greeting in self.greetings):
            return True, "greeting_detected", 1.0
        
        # Check for casual conversation (valid interaction)
        if text_clean in self.casual_phrases or any(phrase in text_clean for phrase in self.casual_phrases):
            return True, "casual_conversation", 0.95
        
        # Check minimum length (but only for non-greeting inputs)
        if len(words) < self.min_length:
            return False, f"It looks like an invalid question. Please provide a complete question.", 0.95  #(minimum {self.min_length} words)

        # Check for single word (unless it's a question word)
        if len(words) == 1:
            if words[0] in self.question_words:
                return False, "Incomplete question. Please provide more details.", 0.9
            else:
                return False, "Single word detected. Please ask a complete question.", 0.95
        
        # Check if it's just repeated characters or gibberish
        if self._is_gibberish(text_clean):
            return False, "It's looks like an invalid question. Please enter a meaningful question about Python.", 0.95
        
        # Scoring system
        score = 0.0
        reasons = []
        
        # 1. Check for question words (high weight)
        has_question_word = any(word in self.question_words for word in words)
        if has_question_word:
            score += 0.4
            reasons.append("question word detected")
        
        # 2. Check for question mark
        if '?' in text:
            score += 0.2
            reasons.append("question mark present")
        
        # 3. Check for programming keywords (essential for Python Q&A)
        programming_words = [word for word in words if word in self.programming_keywords]
        if programming_words:
            score += 0.3
            reasons.append(f"Python-related terms: {', '.join(programming_words[:3])}")
        
        # 4. Check for command patterns (like "how to", "help me")
        for pattern in self.command_patterns:
            if re.search(pattern, text_clean):
                score += 0.3
                reasons.append("command pattern detected")
                break
        
        # 5. Check sentence structure (has verbs and nouns)
        if len(words) >= 4:
            score += 0.1
            reasons.append("complete sentence structure")
        
        # Decision logic
        if score >= 0.5:
            reason_text = "Valid question: " + ", ".join(reasons)
            return True, reason_text, min(score, 1.0)
        elif score >= 0.3:
            # Borderline case - check if it has at least some Python context
            if programming_words or has_question_word:
                return True, "Acceptable question (consider being more specific)", score
            else:
                return False, "Not a clear Python-related question. Please include Python programming terms.", 0.7
        else:
            return False, "This doesn't appear to be a question. Please ask about Python programming concepts, syntax, or problems.", 0.85
    
    def _is_gibberish(self, text: str) -> bool:
        """Detect if text is gibberish (random characters, repeated patterns, etc.)"""
        # Check for repeated characters (like "aaaa", "111")
        if re.search(r'(.)\1{4,}', text):
            return True
        
        # Check for very low vowel-to-consonant ratio (gibberish often has this)
        vowels = len(re.findall(r'[aeiou]', text))
        consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', text))
        
        if consonants > 0:
            vowel_ratio = vowels / (vowels + consonants)
            # English typically has ~40% vowels, gibberish often has <10% or >80%
            if vowel_ratio < 0.1 or vowel_ratio > 0.9:
                return True
        
        # Check if it's just numbers or special characters
        alphanumeric = re.findall(r'[a-z0-9]', text)
        if len(alphanumeric) == 0:
            return True
        
        return False
    
    def get_suggestion(self, text: str) -> str:
        """Provide helpful suggestions when input is invalid"""
        suggestions = [
            "💡 Try asking questions like:",
            "   • 'How do I sort a list in Python?'",
            "   • 'What is a lambda function?'",
            "   • 'Explain list comprehension'",
            "   • 'How to read a file in Python?'",
            "   • 'What is the difference between list and tuple?'",
            "",
            "📝 Tips for better questions:",
            "   • Start with question words (How, What, Why, etc.)",
            "   • Include Python-related terms",
            "   • Be specific about what you want to learn",
            "   • Use complete sentences"
        ]
        return "\n".join(suggestions)


# Test function
def test_validator():
    """Test the question validator"""
    validator = QuestionValidator()
    
    test_cases = [
        "How to sort a list in Python?",
        "What is a lambda function",
        "python",
        "asdfgh",
        "tell me about loops",
        "1234",
        "help",
        "explain list comprehension in python",
        "aaaaaaa",
        "loop",
        "How do I read a file?",
        ""
    ]
    
    print("=" * 80)
    print("QUESTION VALIDATOR TEST")
    print("=" * 80)
    
    for test in test_cases:
        is_valid, reason, confidence = validator.is_valid_question(test)
        print(f"\nInput: '{test}'")
        print(f"Valid: {is_valid}")
        print(f"Reason: {reason}")
        print(f"Confidence: {confidence:.2%}")
        
        if not is_valid:
            print("\n" + validator.get_suggestion(test))
        
        print("-" * 80)


if __name__ == "__main__":
    test_validator()
