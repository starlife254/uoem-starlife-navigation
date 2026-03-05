# active_learning.py
class ActiveLearningSystem:
    def __init__(self, nlp_processor, confidence_threshold: float = 0.7):
        self.nlp_processor = nlp_processor
        self.confidence_threshold = confidence_threshold
        self.uncertain_queries = []
    
    def process_with_uncertainty(self, query: str) -> Dict:
        """Process query and flag uncertain predictions"""
        result = self.nlp_processor.process_query(query)
        
        # Check confidence levels
        if (result['intent_confidence'] < self.confidence_threshold or
            result['building_confidence'] < self.confidence_threshold):
            
            result['needs_feedback'] = True
            self.uncertain_queries.append({
                'query': query,
                'result': result,
                'timestamp': datetime.now()
            })
            
            # Limit queue size
            if len(self.uncertain_queries) > 100:
                self.uncertain_queries.pop(0)
        
        return result
    
    def get_uncertain_samples(self, n: int = 10):
        """Get uncertain samples for human review"""
        return self.uncertain_queries[:n]
    
    def incorporate_feedback(self, feedback_list: List[Dict]):
        """Incorporate human feedback into models"""
        for feedback in feedback_list:
            # Update models based on feedback
            self.update_models_with_feedback(feedback)
        
        # Retrain if enough feedback collected
        if len(feedback_list) >= 20:
            self.trigger_retraining()
    
    def update_models_with_feedback(self, feedback: Dict):
        """Update models incrementally"""
        # Implement online learning or store for batch retraining
        pass