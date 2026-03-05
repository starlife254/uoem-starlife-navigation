# building_recognizer.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class BuildingRecognizer:
    def __init__(self, building_names: List[str]):
        self.building_names = building_names
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.building_embeddings = self.encode_buildings()
    
    def encode_buildings(self):
        """Create embeddings for all buildings"""
        return self.model.encode(self.building_names)
    
    def find_building(self, query: str, threshold: float = 0.6):
        """Find building using semantic similarity"""
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.building_embeddings)[0]
        
        best_match_idx = np.argmax(similarities)
        confidence = similarities[best_match_idx]
        
        if confidence >= threshold:
            return self.building_names[best_match_idx], float(confidence)
        
        return None, 0.0
    
    def add_synonyms(self, building: str, synonyms: List[str]):
        """Add synonyms to improve recognition"""
        building_idx = self.building_names.index(building)
        for synonym in synonyms:
            syn_embedding = self.model.encode([synonym])
            # Average with original embedding
            self.building_embeddings[building_idx] = (
                self.building_embeddings[building_idx] * 0.7 + 
                syn_embedding[0] * 0.3
            )