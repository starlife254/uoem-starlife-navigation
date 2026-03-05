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

# Download NLTK data
# Just check if they exist, don't download at runtime
nltk.data.path.append('/opt/render/nltk_data')  # Add Render's NLTK path
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    # Log a warning but don't try to download
    print("⚠ NLTK data not found - ensure build command downloads it")

class CampusNLPProcessor:
    """NLP processor for understanding campus navigation queries"""
    
    def __init__(self, campus_buildings: List[str]):
        """
        Initialize NLP processor with campus-specific data
        
        Args:
            campus_buildings: List of building names in University of Embu
        """
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except:
            # Create a simple fallback
            self.nlp = None
            print("⚠ spaCy model not available, using fallback")
        
        # Campus-specific data
        self.campus_buildings = campus_buildings
        self.building_synonyms = self._create_building_synonyms()
        
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
        
        # Navigation keywords
        self.navigation_keywords = [
            'near', 'close', 'next', 'between', 'beside', 'opposite',
            'behind', 'front', 'left', 'right', 'north', 'south',
            'east', 'west', 'upstairs', 'downstairs', 'ground', 'floor'
        ]
        
        # Question words
        self.question_words = ['where', 'what', 'when', 'how', 'who', 'which', 'why']
        
        print(f"✅ NLP Processor initialized with {len(campus_buildings)} campus buildings")
    
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
    
    def preprocess_query(self, query: str) -> str:
        """Clean and preprocess the user query"""
        query = query.lower().strip()
        
        # Remove extra spaces
        query = re.sub(r'\s+', ' ', query)
        
        # Remove punctuation except for question marks
        query = re.sub(r'[^\w\s?]', ' ', query)
        
        return query
    
    def extract_intent(self, query: str) -> Tuple[str, float]:
        """
        Extract the intent from user query
        
        Returns:
            Tuple of (intent, confidence_score)
        """
        query = self.preprocess_query(query)
        
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
        query = self.preprocess_query(query)
        
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
                overlap = len(common_words) / len(building_words)
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
    
    def generate_response(self, intent: str, building: Optional[str] = None, 
                         confidence: float = 0.0) -> str:
        """Generate natural language response based on intent"""
        
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
            if confidence > 0.6:
                return f"I'll help you navigate to {building}. Calculating the best route now..."
            else:
                return f"I think you're looking for {building}. Let me find the route for you."
        
        if intent == 'navigation' and not building:
            return "I understand you want navigation help. Which building or location are you looking for?"
        
        if intent == 'information' and building:
            return f"I can provide information about {building}. What would you like to know?"
        
        if intent == 'information' and not building:
            return "What campus location would you like information about?"
        
        if intent == 'unknown':
            return "I'm not sure I understand. Could you rephrase your question? For example: 'Where is the library?' or 'How do I get to the administration building?'"
        
        return "I'll help you with that. Could you provide more details?"
    
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
class SimpleNLPProcessor:
    """Simplified NLP processor for basic functionality"""
    
    def __init__(self, campus_buildings: List[str]):
        self.campus_buildings = [b.lower() for b in campus_buildings]
        
    def process_query(self, query: str) -> Dict:
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