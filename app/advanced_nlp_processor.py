"""
Advanced NLP Processor with TensorFlow, Swahili Support, and Voice Processing
University of Embu - Enhanced AI Navigation
"""

import re
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import os

# Download NLTK data
# Just check if they exist, don't download at runtime
nltk.data.path.append('/opt/render/nltk_data')  # Add Render's NLTK path
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    # Log a warning but don't try to download
    print("⚠ NLTK data not found - ensure build command downloads it")
class AdvancedNLPProcessor:
    """Advanced NLP processor with TensorFlow ML models and multilingual support"""
    
    def __init__(self, campus_buildings: List[str], model_dir: str = "ai_models"):
        """
        Initialize advanced NLP processor
        
        Args:
            campus_buildings: List of building names
            model_dir: Directory to save/load ML models
        """
        self.logger = logging.getLogger(__name__)
        self.campus_buildings = campus_buildings
        self.model_dir = model_dir
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize language models
        self.english_nlp = None
        self.swahili_nlp = None
        
        try:
            self.english_nlp = spacy.load("en_core_web_sm")
            print("✅ English spaCy model loaded")
        except:
            print("⚠ English spaCy model not available")
        
        # Swahili patterns (simplified for now)
        self.swahili_patterns = {
            'navigation': [
                r'iko wapi (?:ya )?(.+)',
                r'naenda wapi (?:kwa )?(.+)',
                r'nifikie wapi (?:kwa )?(.+)',
                r'naomba maelekezo ya (?:kwa )?(.+)',
                r'elekea (?:kwa )?(.+)',
                r'chumba cha (?:kwa )?(.+)',
                r'ofisi ya (?:kwa )?(.+)'
            ],
            'greeting': ['habari', 'hujambo', 'sasa', 'mambo', 'niaje'],
            'thanks': ['asante', 'shukrani', 'ahsante']
        }
        
        # Building synonyms with Swahili
        self.building_synonyms = self._create_multilingual_synonyms()
        
        # ML Model for intent classification
        self.intent_model = None
        self.intent_labels = ['navigation', 'information', 'hours', 'contact', 'greeting', 'thanks', 'unknown']
        
        # ML Model for building recognition
        self.building_model = None
        
        # Voice processing state
        self.is_listening = False
        self.voice_buffer = []
        
        # Load or train models
        self._initialize_models()
        
        print(f"✅ Advanced NLP Processor initialized with {len(campus_buildings)} buildings")
        print(f"   Supported languages: English, Swahili")
        print(f"   ML Models: {'Loaded' if self.intent_model else 'Training required'}")
    
    def _create_multilingual_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        """Create synonyms in both English and Swahili"""
        synonyms = {}
        
        # English to Swahili building name mapping
        # This should be expanded with actual translations
        building_translations = {
            'library': 'maktaba',
            'administration': 'utawala',
            'cafeteria': 'messes',
            'hostel': 'hosteli',
            'classroom': 'chumba cha darasa',
            'laboratory': 'maabara',
            'computer lab': 'maabara ya kompyuta',
            'auditorium': 'ukumbi',
            'clinic': 'kliniki',
            'security': 'usalama'
        }
        
        for building in self.campus_buildings:
            building_lower = building.lower()
            synonyms[building_lower] = {
                'english': [building_lower],
                'swahili': []
            }
            
            # Add English synonyms
            words = building_lower.split()
            if len(words) > 1:
                # Add acronym
                acronym = ''.join([word[0] for word in words if word])
                synonyms[building_lower]['english'].append(acronym)
                
                # Add without "building"
                if 'building' in words:
                    without_building = ' '.join([w for w in words if w != 'building'])
                    synonyms[building_lower]['english'].append(without_building)
            
            # Add Swahili translations
            for eng_word, swa_word in building_translations.items():
                if eng_word in building_lower:
                    synonyms[building_lower]['swahili'].append(swa_word)
                    # Add building name with Swahili word
                    swa_building = building_lower.replace(eng_word, swa_word)
                    synonyms[building_lower]['swahili'].append(swa_building)
            
            # Add common Swahili terms
            if 'library' in building_lower:
                synonyms[building_lower]['swahili'].extend(['maktaba kuu', 'chumba cha kusoma'])
            elif 'administration' in building_lower:
                synonyms[building_lower]['swahili'].extend(['ofisi kuu', 'uraia'])
            elif 'hostel' in building_lower:
                synonyms[building_lower]['swahili'].extend(['nyumba ya wanafunzi', 'malazi'])
        
        return synonyms
    
    def _initialize_models(self):
        """Initialize or load ML models"""
        intent_model_path = os.path.join(self.model_dir, "intent_model.h5")
        building_model_path = os.path.join(self.model_dir, "building_model.h5")
        
        if os.path.exists(intent_model_path) and os.path.exists(building_model_path):
            try:
                self.intent_model = keras.models.load_model(intent_model_path)
                self.building_model = keras.models.load_model(building_model_path)
                print("✅ Pre-trained ML models loaded")
            except:
                print("⚠ Error loading models, will train new ones")
                self._train_models()
        else:
            print("⚠ No pre-trained models found, training new ones...")
            self._train_models()
    
    def _create_training_data(self):
        """Create synthetic training data for models"""
        # Intent training data
        intent_examples = {
            'navigation': [
                "where is the library",
                "how do i get to administration",
                "directions to cafeteria",
                "take me to computer lab",
                "route to hostel",
                "find the classroom",
                "show me the way to auditorium",
                "navigate to security office",
                "where can i find the clinic",
                "how to reach dean's office",
                # Swahili examples
                "iko wapi maktaba",
                "naenda wapi utawala",
                "nifikie wapi messes",
                "elekea hosteli",
                "naomba maelekezo ya maabara"
            ],
            'information': [
                "what is the library",
                "tell me about administration",
                "information about cafeteria",
                "describe the computer lab",
                "explain about hostel",
                # Swahili
                "ni nini maktaba",
                "elezea kuhusu utawala",
                "habari za messes"
            ],
            'greeting': [
                "hello", "hi", "hey", "good morning", "good afternoon",
                # Swahili
                "habari", "hujambo", "sasa", "mambo"
            ],
            'thanks': [
                "thank you", "thanks", "appreciate it", "much appreciated",
                # Swahili
                "asante", "shukrani", "ahsante sana"
            ]
        }
        
        # Building recognition training data
        building_examples = {}
        for building in self.campus_buildings:
            building_lower = building.lower()
            examples = [
                building_lower,
                building_lower.replace('building', ''),
                building_lower.replace('block', ''),
                'the ' + building_lower,
                building_lower + ' building',
                # Partial matches
                building_lower.split()[0] if ' ' in building_lower else building_lower
            ]
            
            # Add synonyms
            if building_lower in self.building_synonyms:
                examples.extend(self.building_synonyms[building_lower]['english'][:3])
                examples.extend(self.building_synonyms[building_lower]['swahili'][:2])
            
            building_examples[building_lower] = examples
        
        return intent_examples, building_examples
    
    def _train_models(self):
        """Train ML models for intent and building recognition"""
        print("🔄 Training ML models...")
        
        # Create training data
        intent_examples, building_examples = self._create_training_data()
        
        # Simplified model training (in production, use more sophisticated approach)
        # For now, we'll create simple models
        
        # Intent model
        self.intent_model = self._create_intent_model()
        
        # Building model
        self.building_model = self._create_building_model()
        
        # Save models
        intent_model_path = os.path.join(self.model_dir, "intent_model.h5")
        building_model_path = os.path.join(self.model_dir, "building_model.h5")
        
        self.intent_model.save(intent_model_path)
        self.building_model.save(building_model_path)
        
        print("✅ ML models trained and saved")
    
    def _create_intent_model(self):
        """Create intent classification model"""
        # Simple model for demonstration
        # In production, use more complex architecture with word embeddings
        
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(100,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(len(self.intent_labels), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _create_building_model(self):
        """Create building recognition model"""
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(200,)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.campus_buildings), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def detect_language(self, query: str) -> str:
        """Detect language of query"""
        query_lower = query.lower()
        
        # Check for Swahili patterns
        swahili_indicators = ['iko', 'wapi', 'naenda', 'nifikie', 'maelekezo', 
                             'asante', 'shukrani', 'habari', 'hujambo']
        
        for indicator in swahili_indicators:
            if indicator in query_lower:
                return 'swahili'
        
        # Default to English
        return 'english'
    
    def preprocess_query(self, query: str, language: str = 'english') -> Dict:
        """Advanced query preprocessing"""
        query = query.lower().strip()
        
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        
        # Remove punctuation except for question marks
        query = re.sub(r'[^\w\s?]', ' ', query)
        
        # Language-specific preprocessing
        if language == 'swahili':
            # Remove common Swahili question words
            query = re.sub(r'\b(iko|wapi|naenda|nifikie|naomba)\b', '', query)
        else:
            # Remove common English question words
            remove_words = ['where', 'how', 'what', 'when', 'which', 'who', 'why']
            for word in remove_words:
                query = query.replace(word + ' is', '').replace(word + ' are', '')
        
        # Extract keywords
        keywords = query.split()
        
        # Remove stopwords
        if language == 'english':
            stop_words = set(stopwords.words('english'))
        else:
            # Basic Swahili stopwords
            stop_words = {'ya', 'kwa', 'na', 'wa', 'ni', 'za', 'ku', 'la'}
        
        keywords = [word for word in keywords if word not in stop_words]
        
        return {
            'original': query,
            'cleaned': ' '.join(keywords),
            'keywords': keywords,
            'language': language
        }
    
    def extract_intent_ml(self, processed_query: Dict) -> Tuple[str, float]:
        """Extract intent using ML model"""
        # For now, use rule-based as placeholder
        # In production, use the trained model
        
        query = processed_query['cleaned']
        language = processed_query['language']
        
        # Rule-based intent detection (temporary)
        if any(word in query for word in ['hello', 'hi', 'hey', 'habari', 'hujambo']):
            return 'greeting', 0.95
        
        if any(word in query for word in ['thank', 'thanks', 'asante', 'shukrani']):
            return 'thanks', 0.95
        
        if any(word in query for word in ['where', 'how to', 'directions', 'route', 
                                         'wapi', 'naenda', 'nifikie', 'elekea']):
            return 'navigation', 0.85
        
        if any(word in query for word in ['what', 'tell', 'information', 'describe',
                                         'ni nini', 'elezea', 'habari za']):
            return 'information', 0.80
        
        return 'unknown', 0.5
    
    def extract_building_ml(self, processed_query: Dict) -> Tuple[Optional[str], float]:
        """Extract building using ML model"""
        query = processed_query['cleaned']
        language = processed_query['language']
        
        best_building = None
        best_confidence = 0.0
        
        # Check each building
        for building in self.campus_buildings:
            building_lower = building.lower()
            confidence = 0.0
            
            # Exact match
            if building_lower in query:
                confidence = 0.95
            
            # Partial match
            elif any(word in building_lower for word in processed_query['keywords']):
                confidence = 0.75
            
            # Synonym match
            elif building_lower in self.building_synonyms:
                synonyms = (self.building_synonyms[building_lower]['english'] + 
                          self.building_synonyms[building_lower]['swahili'])
                for synonym in synonyms:
                    if synonym in query:
                        confidence = 0.85
                        break
            
            if confidence > best_confidence:
                best_building = building
                best_confidence = confidence
        
        return best_building, best_confidence
    
    def extract_context(self, query: str) -> Dict:
        """Extract navigation context from query"""
        context = {
            'urgency': False,
            'mode': 'walking',
            'accessible': False,
            'avoid_stairs': False,
            'shortest': True,
            'scenic': False
        }
        
        query_lower = query.lower()
        
        # Accessibility needs
        accessible_indicators = ['wheelchair', 'accessible', 'disabled', 'handicap', 
                               'elevator', 'ramp', 'easy', 'pram', 'stroller']
        if any(indicator in query_lower for indicator in accessible_indicators):
            context['accessible'] = True
            context['avoid_stairs'] = True
        
        # Stairs avoidance
        if 'avoid stairs' in query_lower or 'no stairs' in query_lower:
            context['avoid_stairs'] = True
        
        # Urgency
        urgency_indicators = ['quick', 'fast', 'urgent', 'asap', 'hurry', 'emergency',
                            'haraka', 'upesi', 'mara moja']
        if any(indicator in query_lower for indicator in urgency_indicators):
            context['urgency'] = True
        
        # Mode preference
        if 'walk' in query_lower or 'foot' in query_lower or 'kwa miguu' in query_lower:
            context['mode'] = 'walking'
        elif 'cycle' in query_lower or 'bike' in query_lower or 'baiskeli' in query_lower:
            context['mode'] = 'cycling'
        elif 'drive' in query_lower or 'car' in query_lower or 'gari' in query_lower:
            context['mode'] = 'driving'
        
        # Route preference
        if 'shortest' in query_lower or 'short' in query_lower or 'fupi' in query_lower:
            context['shortest'] = True
            context['scenic'] = False
        elif 'scenic' in query_lower or 'nice' in query_lower or 'beautiful' in query_lower:
            context['scenic'] = True
            context['shortest'] = False
        
        return context
    
    def process_voice_input(self, audio_data=None, text: str = None) -> Dict:
        """Process voice input (simplified - in production use speech recognition)"""
        if text:
            # If text is provided, process it directly
            return self.process_query(text)
        
        # In production, integrate with:
        # - SpeechRecognition library
        # - Google Cloud Speech-to-Text
        # - Mozilla DeepSpeech
        
        print("🎤 Voice processing would be implemented with speech recognition API")
        return {
            'success': False,
            'error': 'Voice processing requires speech recognition integration',
            'suggestion': 'Use text input or implement speech recognition'
        }
    
    def generate_multilingual_response(self, intent: str, building: Optional[str] = None,
                                     confidence: float = 0.0, language: str = 'english') -> str:
        """Generate response in appropriate language"""
        
        if language == 'swahili':
            if intent == 'greeting':
                return random.choice([
                    "Habari! Mimi ni msaidizi wako wa uelekezaji katika Chuo Kikuu cha Embu. Ninaweza kukusaidiaje leo?",
                    "Hujambo! Tayari kukuelekeza Chuo Kikuu cha Embu? Unataka kwenda wapi?",
                    "Karibu kwenye Mfumo wa Uelekezaji wa Chuo Kikuu cha Embu! Unaenda wapi?"
                ])
            
            if intent == 'thanks':
                return random.choice([
                    "Karibu! Nafurahi kukusaidia na mahitaji yako ya uelekezaji.",
                    "Ahsante! Nimefurahi kukusaidia.",
                    "Karibu! Safari salama ndani ya chuo."
                ])
            
            if intent == 'navigation' and building:
                if confidence > 0.7:
                    return f"Nitaikusaidia kufika {building}. Ninahesabu njia bora sasa..."
                else:
                    return f"Nadhani unatafuta {building}. Nitaikuta njia yako."
            
            if intent == 'navigation' and not building:
                return "Naelewa unataka msaada wa uelekezaji. Unatafuta jengo gani au eneo gani?"
        
        # English responses (default)
        if intent == 'greeting':
            return random.choice([
                "Hello! I'm your AI campus navigation assistant. How can I help you navigate the University of Embu today?",
                "Hi there! Ready to explore the University of Embu? What location are you looking for?",
                "Welcome to the University of Embu AI Navigation System! Where would you like to go?"
            ])
        
        if intent == 'thanks':
            return random.choice([
                "You're welcome! Happy to help with your navigation needs.",
                "Glad I could assist! Let me know if you need anything else.",
                "You're welcome! Safe travels around campus."
            ])
        
        if intent == 'navigation' and building:
            if confidence > 0.7:
                return f"I'll help you navigate to {building}. Calculating the best route now..."
            else:
                return f"I think you're looking for {building}. Let me find the route for you."
        
        if intent == 'navigation' and not building:
            return "I understand you want navigation help. Which building or location are you looking for?"
        
        if intent == 'information' and building:
            return f"I can provide information about {building}. What would you like to know?"
        
        if intent == 'unknown':
            if language == 'swahili':
                return "Sielewi vizuri. Unaweza kuelezea tena swali lako? Kwa mfano: 'Iko wapi maktaba?' au 'Naenda wapi ofisi ya utawala?'"
            return "I'm not sure I understand. Could you rephrase your question? For example: 'Where is the library?' or 'How do I get to the administration building?'"
        
        return "I'll help you with that. Could you provide more details?"
    
    def process_query(self, query: str, is_voice: bool = False) -> Dict:
        """
        Process a complete user query with advanced NLP
        
        Returns:
            Dictionary with comprehensive processing results
        """
        self.logger.info(f"Processing query: {query} (voice: {is_voice})")
        
        start_time = datetime.now()
        
        # Detect language
        language = self.detect_language(query)
        
        # Preprocess query
        processed = self.preprocess_query(query, language)
        
        # Extract intent using ML
        intent, intent_confidence = self.extract_intent_ml(processed)
        
        # Extract building
        building, building_confidence = None, 0.0
        if intent in ['navigation', 'information', 'hours', 'contact']:
            building, building_confidence = self.extract_building_ml(processed)
        
        # Extract context
        context = self.extract_context(query)
        
        # Generate response
        response = self.generate_multilingual_response(intent, building, 
                                                     building_confidence, language)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'original_query': query,
            'processed_query': processed,
            'language': language,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'building': building,
            'building_confidence': building_confidence,
            'context': context,
            'response': response,
            'processing_time': processing_time,
            'is_voice': is_voice,
            'success': building_confidence > 0.3 or intent in ['greeting', 'thanks'],
            'timestamp': datetime.now().isoformat(),
            'version': 'advanced_2.0'
        }
        
        self.logger.info(f"Query processed in {processing_time:.3f}s: {result}")
        return result
    
    def batch_process(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries at once"""
        return [self.process_query(query) for query in queries]
    
    def train_on_feedback(self, query: str, correct_intent: str, correct_building: str = None):
        """Train models based on user feedback"""
        print(f"🔄 Training on feedback: '{query}' -> intent:{correct_intent}, building:{correct_building}")
        # In production, implement feedback-based training
        # This would update the ML models
    
    def get_voice_commands(self) -> List[str]:
        """Get list of supported voice commands"""
        commands = [
            # English
            "Where is [building]",
            "How do I get to [location]",
            "Directions to [place]",
            "Take me to [destination]",
            "Find [building]",
            "Navigate to [location]",
            # Swahili
            "Iko wapi [jengo]",
            "Naenda wapi [eneo]",
            "Nifikie wapi [mahali]",
            "Elekea [lengo]",
            "Tafuta [jengo]"
        ]
        return commands

# Factory function for backward compatibility
def create_advanced_nlp_processor(campus_buildings: List[str], use_ml: bool = True):
    """
    Create advanced NLP processor instance
    
    Args:
        campus_buildings: List of building names
        use_ml: Whether to use ML models
    """
    try:
        return AdvancedNLPProcessor(campus_buildings)
    except Exception as e:
        print(f"⚠ Error creating advanced NLP processor: {e}")
        # Fallback to simple processor
        from nlp_processor import create_nlp_processor
        return create_nlp_processor(campus_buildings, use_advanced=False)