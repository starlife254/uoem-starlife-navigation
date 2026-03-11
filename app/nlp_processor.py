"""
NLP Processor for AI-Aided Campus Navigation System
University of Embu - Implementation for Research Proposal
"""

import re
import json
import random
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Dict, List, Tuple, Optional
import logging
import sys
import os
import threading
import time
import importlib.util

print("🔍 DEBUG: nlp_processor.py file loaded", file=sys.stderr)
print("🔍 DEBUG: Starting to import nltk in nlp_processor.py", file=sys.stderr)

# Verify spaCy model installation - SAFE VERSION without subprocess
print("🔍 DEBUG: Checking spaCy model installation safely...", file=sys.stderr)

# Simple and reliable check - try to import the model directly
spacy_model_available = False
try:
    # Method 1: Try to load with spacy
    import spacy
    print(f"🔍 DEBUG: spaCy version: {spacy.__version__}", file=sys.stderr)
    
    # Try a quick, non-blocking check
    model_spec = importlib.util.find_spec("en_core_web_sm")
    if model_spec is not None:
        print("✅ en_core_web_sm module found via find_spec", file=sys.stderr)
        spacy_model_available = True
    else:
        # Fallback: try a quick load with timeout protection
        model_loaded = [False]
        
        def load_model():
            try:
                nlp = spacy.load("en_core_web_sm")
                model_loaded[0] = True
            except:
                pass
        
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()
        thread.join(timeout=2)  # Wait max 2 seconds
        
        if model_loaded[0]:
            print("✅ spaCy model loaded successfully (quick check)", file=sys.stderr)
            spacy_model_available = True
        else:
            print("⚠ spaCy model load timed out or failed", file=sys.stderr)
            spacy_model_available = False
            
except Exception as e:
    print(f"⚠ Error checking spaCy model: {e}", file=sys.stderr)
    spacy_model_available = False

print(f"🔍 DEBUG: spaCy model available: {spacy_model_available}", file=sys.stderr)

# Set NLTK data path to a persistent location
nltk_data_dir = '/opt/render/nltk_data'
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Also add the current directory as fallback
nltk.data.path.append(os.path.join(os.getcwd(), 'nltk_data'))

print(f"🔍 NLTK data paths: {nltk.data.path}", file=sys.stderr)

# Check if data exists, but DON'T crash if it doesn't - we'll handle gracefully
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK punkt found", file=sys.stderr)
except LookupError:
    print("⚠ NLTK punkt not found - will use fallback methods", file=sys.stderr)

try:
    nltk.data.find('corpora/stopwords')
    print("✅ NLTK stopwords found", file=sys.stderr)
except LookupError:
    print("⚠ NLTK stopwords not found - will use fallback methods", file=sys.stderr)

print("🔍 DEBUG: nltk imports complete", file=sys.stderr)


class CampusNLPProcessor:
    """NLP processor for understanding campus navigation queries"""

    def __init__(self, campus_buildings: List[str]):
        """
        Initialize NLP processor with campus-specific data

        Args:
            campus_buildings: List of building names in University of Embu
        """
        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 1", file=sys.stderr)
        self.logger = logging.getLogger(__name__)

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 2 - loading spaCy", file=sys.stderr)
        self.nlp = None
        if spacy_model_available:
            try:
                # Use the global flag we set above
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy model loaded successfully in __init__", file=sys.stderr)
            except Exception as e:
                print(f"⚠ Could not load spaCy model in __init__: {e}", file=sys.stderr)
                # Try fallback import
                try:
                    import en_core_web_sm
                    self.nlp = en_core_web_sm.load()
                    print("✅ spaCy model loaded via en_core_web_sm.load()", file=sys.stderr)
                except Exception as e2:
                    print(f"⚠ spaCy model also failed via direct import: {e2}", file=sys.stderr)
                    self.nlp = None
        else:
            print("⚠ spaCy model not available (from pre-check), using fallback NLP", file=sys.stderr)

        if self.nlp is None:
            print("⚠ spaCy model not available, using fallback NLP", file=sys.stderr)

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 3 - setting campus buildings", file=sys.stderr)
        self.campus_buildings = campus_buildings

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 4 - creating synonyms", file=sys.stderr)
        self.building_synonyms = self._create_building_synonyms()

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 5 - setting up intent patterns", file=sys.stderr)
        # Intent patterns
        self.intent_patterns = {
            'navigation': [
                r'where is (?:the )?(.+)',
                r'how do i get to (?:the )?(.+)',
                r'directions to (?:the )?(.+)',
                r'route to (?:the )?(.+)',
                r'go to (?:the )?(.+)',
                r'take me to (?:the )?(.+)',
                r'navigate to (?:the )?(.+)',
                r'location of (?:the )?(.+)',
                r'find (?:the )?(.+)',
                r'show me (?:the )?(.+)'
            ],
            'information': [
                r'what is (?:the )?(.+)',
                r'tell me about (?:the )?(.+)',
                r'information about (?:the )?(.+)',
                r'describe (?:the )?(.+)',
                r'explain (?:the )?(.+)',
                r'what are (?:the )?(.+)',
                r'details of (?:the )?(.+)'
            ],
            'hours': [
                r'when does (?:the )?(.+) open',
                r'opening hours of (?:the )?(.+)',
                r'closing time of (?:the )?(.+)',
                r'what time does (?:the )?(.+) close',
                r'operating hours of (?:the )?(.+)'
            ],
            'contact': [
                r'contact (?:the )?(.+)',
                r'phone number of (?:the )?(.+)',
                r'email of (?:the )?(.+)',
                r'how to contact (?:the )?(.+)',
                r'who is in charge of (?:the )?(.+)'
            ],
            'facilities': [
                r'facilities in (?:the )?(.+)',
                r'what does (?:the )?(.+) have',
                r'amenities in (?:the )?(.+)',
                r'equipment in (?:the )?(.+)',
                r'services in (?:the )?(.+)'
            ]
        }

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 6 - setting up navigation keywords", file=sys.stderr)
        # Navigation keywords
        self.navigation_keywords = [
            'near', 'close', 'next', 'between', 'beside', 'opposite',
            'behind', 'front', 'left', 'right', 'north', 'south',
            'east', 'west', 'upstairs', 'downstairs', 'ground', 'floor'
        ]

        print("🔍 DEBUG: CampusNLPProcessor.__init__ - step 7 - setting up question words", file=sys.stderr)
        # Question words
        self.question_words = ['where', 'what', 'when', 'how', 'who', 'which', 'why']

        print(f"✅ NLP Processor initialized with {len(campus_buildings)} campus buildings", file=sys.stderr)
        print("🔍 DEBUG: CampusNLPProcessor.__init__ completed successfully", file=sys.stderr)

    def _create_building_synonyms(self) -> Dict[str, List[str]]:
        """Create synonyms and common names for campus buildings"""
        synonyms = {}

        for building in self.campus_buildings:
            building_lower = building.lower()
            synonyms[building_lower] = [building_lower]

            # Add common abbreviations and variations
            words = building_lower.split()
            if len(words) > 1:
                # Add acronym
                acronym = ''.join([word[0] for word in words if word])
                synonyms[building_lower].append(acronym)

                # Add building without "building" suffix
                if 'building' in words:
                    without_building = ' '.join([w for w in words if w != 'building'])
                    synonyms[building_lower].append(without_building)

                # Add with "block" instead of "building"
                if 'building' in words:
                    with_block = building_lower.replace('building', 'block')
                    synonyms[building_lower].append(with_block)

            # Common building types
            building_types = {
                'library': ['lib', 'book house', 'reading area'],
                'hostel': ['dorm', 'dormitory', 'residence', 'accommodation'],
                'cafeteria': ['canteen', 'mess', 'dining', 'food court'],
                'auditorium': ['hall', 'main hall', 'conference hall'],
                'administration': ['admin', 'administrative block', 'registry'],
                'laboratory': ['lab', 'science lab', 'research lab'],
                'classroom': ['lecture room', 'class', 'tutorial room'],
                'office': ['staff room', 'faculty room', 'department office']
            }

            for btype, syns in building_types.items():
                if btype in building_lower:
                    synonyms[building_lower].extend(syns)

        return synonyms

    def preprocess_query(self, query: str, language: str = 'english') -> Dict:
        """Clean and preprocess the user query"""
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

        # Remove stopwords - with fallback if NLTK fails
        try:
            if language == 'english':
                stop_words = set(stopwords.words('english'))
            else:
                # Basic Swahili stopwords
                stop_words = {'ya', 'kwa', 'na', 'wa', 'ni', 'za', 'ku', 'la'}
        except LookupError:
            # Fallback if NLTK data missing
            stop_words = {'the', 'a', 'an', 'is', 'at', 'which', 'on', 'for', 'in', 'to', 'and'}
            print("⚠ Using fallback stopwords in nlp_processor.py", file=sys.stderr)

        keywords = [word for word in keywords if word not in stop_words]

        return {
            'original': query,
            'cleaned': ' '.join(keywords),
            'keywords': keywords,
            'language': language
        }

    def extract_intent(self, query: str) -> Tuple[str, float]:
        """
        Extract the intent from user query

        Returns:
            Tuple of (intent, confidence_score)
        """
        query = self.preprocess_query(query)['cleaned']

        # Check for greeting
        if any(greeting in query for greeting in ['hello', 'hi', 'hey', 'greetings']):
            return 'greeting', 1.0

        # Check for thanks
        if any(thanks in query for thanks in ['thank', 'thanks', 'appreciate']):
            return 'thanks', 1.0

        # Pattern matching for intents
        best_intent = 'unknown'
        best_confidence = 0.0

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    confidence = min(1.0, len(pattern) / 100)  # Simple confidence scoring
                    if confidence > best_confidence:
                        best_intent = intent
                        best_confidence = confidence

        # If no pattern matched but has question word
        if best_intent == 'unknown':
            if any(word in query.split() for word in self.question_words):
                if 'where' in query:
                    return 'navigation', 0.7
                else:
                    return 'information', 0.6

        return best_intent, best_confidence

    def extract_building_name(self, query: str) -> Tuple[Optional[str], float]:
        """
        Extract building name from query

        Returns:
            Tuple of (building_name, confidence_score)
        """
        query = self.preprocess_query(query)['cleaned']

        # Remove common phrases
        remove_phrases = [
            'where is', 'how do i get to', 'directions to', 'route to',
            'go to', 'take me to', 'navigate to', 'location of',
            'find', 'show me', 'what is', 'tell me about'
        ]

        for phrase in remove_phrases:
            query = query.replace(phrase, '').strip()

        # Try exact match first
        for building in self.campus_buildings:
            building_lower = building.lower()
            if building_lower in query:
                return building, 0.9

        # Try synonym match
        for building, synonyms in self.building_synonyms.items():
            for synonym in synonyms:
                if synonym in query and len(synonym) > 2:
                    # Find original building name
                    for orig_building in self.campus_buildings:
                        if building in orig_building.lower():
                            return orig_building, 0.8

        # Try partial match
        query_words = set(query.split())
        for building in self.campus_buildings:
            building_words = set(building.lower().split())
            common_words = query_words.intersection(building_words)
            if len(common_words) >= 1:
                # Calculate overlap ratio
                overlap = len(common_words) / len(building_words) if building_words else 0
                if overlap >= 0.3:  # At least 30% overlap
                    return building, overlap

        # Use spaCy for named entity recognition if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'FAC', 'LOC']:
                    # Check if it might be a building
                    for building in self.campus_buildings:
                        if ent.text.lower() in building.lower():
                            return building, 0.7

        return None, 0.0

    def extract_navigation_context(self, query: str) -> Dict:
        """Extract navigation context from query"""
        context = {
            'urgency': False,
            'mode': 'walking',
            'avoid_crowds': False,
            'accessible': False
        }

        query_lower = query.lower()

        # Check for urgency
        urgency_indicators = ['quick', 'fast', 'urgent', 'asap', 'hurry', 'emergency']
        if any(indicator in query_lower for indicator in urgency_indicators):
            context['urgency'] = True

        # Check for mode preference
        if 'walk' in query_lower or 'on foot' in query_lower:
            context['mode'] = 'walking'
        elif 'cycle' in query_lower or 'bike' in query_lower or 'bicycle' in query_lower:
            context['mode'] = 'cycling'
        elif 'drive' in query_lower or 'car' in query_lower or 'vehicle' in query_lower:
            context['mode'] = 'driving'

        # Check for accessibility needs
        accessibility_indicators = ['wheelchair', 'accessible', 'disabled', 'handicap', 'elevator']
        if any(indicator in query_lower for indicator in accessibility_indicators):
            context['accessible'] = True

        # Check for crowd avoidance
        crowd_indicators = ['crowd', 'busy', 'congested', 'quiet', 'less people']
        if any(indicator in query_lower for indicator in crowd_indicators):
            context['avoid_crowds'] = True

        return context

    def generate_response(self, intent: str, building: Optional[str] = None, confidence: float = 0.0) -> str:
        """Generate natural, friendly responses"""
        if intent == 'greeting':
            return random.choice([
                "👋 Hello! I'm Starlife AI your campus navigation assistant. Where would you like to go today?",
                "Hi there! 🌟 Ready to explore the University of Embu? Just tell me which building you're looking for!",
                "Welcome! 🎓 I can help you find any building on campus. What are you looking for?"
            ])

        if intent == 'thanks':
            return random.choice([
                "You're very welcome! 😊 Let me know if you need anything else!",
                "Happy to help! 🗺️ Enjoy your time on campus!",
                "Anytime! 👍 Safe travels around the university!"
            ])

        if intent == 'navigation' and building:
            if confidence > 0.7:
                return f"📍 I'll help you get to **{building}**! Let me calculate the best route for you... 🚶"
            else:
                return f"I think you're looking for **{building}**. Let me find the best way to get you there! 🔍"

        if intent == 'navigation' and not building:
            return "I'd be happy to give you directions! 🗺️ Which building or location are you trying to reach?"

        if intent == 'information' and building:
            return f"📚 Sure! I can tell you about **{building}**. What would you like to know? (opening hours, facilities, contact info, etc.)"

        if intent == 'unknown':
            return "Hmm, I'm not quite sure I understood. 🤔 Could you rephrase that? For example, try asking: 'Where is the library?' or 'How do I get to the administration block?'"

        return "I'll do my best to help! Could you give me a bit more detail about what you're looking for? 🤗"

    def process_query(self, query: str) -> Dict:
        """
        Process a complete user query

        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing query: {query}")

        # Extract intent
        intent, intent_confidence = self.extract_intent(query)

        # Extract building name if relevant
        building = None
        building_confidence = 0.0
        if intent in ['navigation', 'information', 'hours', 'contact', 'facilities']:
            building, building_confidence = self.extract_building_name(query)

        # Extract navigation context
        context = self.extract_navigation_context(query)

        # Generate response
        response = self.generate_response(intent, building, building_confidence)

        result = {
            'original_query': query,
            'intent': intent,
            'intent_confidence': intent_confidence,
            'building': building,
            'building_confidence': building_confidence,
            'context': context,
            'response': response,
            'success': building_confidence > 0.3 or intent in ['greeting', 'thanks']
        }

        self.logger.info(f"Query processed: {result}")
        return result


# Simple implementation for testing

    """Simplified NLP processor for basic functionality"""
class SimpleNLPProcessor:
    """Simplified NLP processor for basic functionality"""
    
    def __init__(self, campus_buildings: List[str]):
        self.campus_buildings = [b.lower() for b in campus_buildings]
    
    def process_query(self, query: str, is_voice: bool = False) -> Dict:
        """Process query with optional voice parameter"""
        # Your existing code...
        query_lower = query.lower()

        # Check for building names
        found_building = None
        for building in self.campus_buildings:
            if building in query_lower:
                found_building = building.title()
                break

        if found_building:
            return {
                'original_query': query,
                'intent': 'navigation',
                'intent_confidence': 0.8,
                'building': found_building,
                'building_confidence': 0.7,
                'response': f"I found {found_building} in your query. Let me help you navigate there.",
                'success': True
            }

        # Check for greetings
        greetings = ['hello', 'hi', 'hey']
        if any(g in query_lower for g in greetings):
            return {
                'original_query': query,
                'intent': 'greeting',
                'intent_confidence': 1.0,
                'building': None,
                'building_confidence': 0.0,
                'response': "Hello! I can help you navigate the University of Embu. Where would you like to go?",
                'success': True
            }

        return {
            'original_query': query,
            'intent': 'unknown',
            'intent_confidence': 0.0,
            'building': None,
            'building_confidence': 0.0,
            'response': "I'm not sure what you're looking for. Try asking about a specific building like 'library' or 'administration'.",
            'success': False
        }


# Factory function to create appropriate processor
def create_nlp_processor(campus_buildings: List[str], use_advanced: bool = True):
    """
    Create an NLP processor instance

    Args:
        campus_buildings: List of building names
        use_advanced: Whether to use advanced NLP (spaCy) or simple version
    """
    try:
        if use_advanced:
            return CampusNLPProcessor(campus_buildings)
        else:
            return SimpleNLPProcessor(campus_buildings)
    except Exception as e:
        print(f"⚠ Error creating advanced NLP processor, using simple version: {e}")
        return SimpleNLPProcessor(campus_buildings)