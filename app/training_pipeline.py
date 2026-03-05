# training_pipeline.py
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import joblib

class AITrainingPipeline:
    def __init__(self, data_path: str = 'training_data'):
        self.data_path = data_path
        self.intent_classifier = None
        self.building_recognizer = None
    
    def load_training_data(self):
        """Load and prepare training data"""
        # Load synthetic data
        with open(f'{self.data_path}/synthetic_queries.json', 'r') as f:
            synthetic_data = json.load(f)
        
        # Load real user data from feedback database
        conn = sqlite3.connect('ai_feedback.db')
        real_data = pd.read_sql_query('SELECT * FROM user_queries', conn)
        conn.close()
        
        # Combine data
        training_data = self.combine_data(synthetic_data, real_data)
        
        return training_data
    
    def combine_data(self, synthetic_data, real_data):
        """Combine synthetic and real data"""
        combined = []
        
        # Add synthetic data
        for item in synthetic_data:
            combined.append({
                'text': item['query'],
                'intent': item['intent'],
                'building': item['building'],
                'language': item.get('language', 'english')
            })
        
        # Add real data (successful queries)
        for _, row in real_data[real_data['success'] == 1].iterrows():
            combined.append({
                'text': row['query'],
                'intent': row['intent_detected'],
                'building': row['building_detected'],
                'language': row['language']
            })
        
        return pd.DataFrame(combined)
    
    def train_intent_classifier(self, training_data):
        """Train intent classification model"""
        # Preprocess data
        X = training_data['text'].tolist()
        y = pd.get_dummies(training_data['intent'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Choose model type based on data size
        if len(training_data) > 1000:
            classifier = AdvancedIntentClassifier(
                num_intents=len(y.columns),
                model_type='bert'
            )
        else:
            classifier = AdvancedIntentClassifier(
                num_intents=len(y.columns),
                model_type='cnn'
            )
        
        # Train
        if classifier.model_type == 'bert':
            X_train_ids, X_train_mask = classifier.preprocess_bert(X_train)
            X_test_ids, X_test_mask = classifier.preprocess_bert(X_test)
            
            history = classifier.model.fit(
                [X_train_ids, X_train_mask],
                y_train,
                validation_data=([X_test_ids, X_test_mask], y_test),
                epochs=10,
                batch_size=16
            )
        else:
            # Tokenize for CNN
            tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(X_train)
            
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            
            X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
                X_train_seq, maxlen=100
            )
            X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(
                X_test_seq, maxlen=100
            )
            
            history = classifier.model.fit(
                X_train_pad,
                y_train,
                validation_data=(X_test_pad, y_test),
                epochs=20,
                batch_size=32
            )
        
        self.intent_classifier = classifier
        return history
    
    def train_building_recognizer(self, training_data):
        """Train building recognition model"""
        building_names = training_data['building'].unique().tolist()
        recognizer = BuildingRecognizer(building_names)
        
        # Add synonyms from training data
        for building in building_names:
            synonyms = training_data[
                training_data['building'] == building
            ]['text'].tolist()[:5]  # Use first 5 queries as synonyms
            recognizer.add_synonyms(building, synonyms)
        
        self.building_recognizer = recognizer
    
    def evaluate_models(self, test_data):
        """Evaluate model performance"""
        results = {}
        
        # Intent classification evaluation
        if self.intent_classifier:
            # Calculate accuracy, precision, recall, F1-score
            pass
        
        # Building recognition evaluation
        if self.building_recognizer:
            correct = 0
            total = len(test_data)
            
            for _, row in test_data.iterrows():
                predicted, confidence = self.building_recognizer.find_building(
                    row['text']
                )
                if predicted == row['building']:
                    correct += 1
            
            results['building_accuracy'] = correct / total
        
        return results
    
    def save_models(self, output_dir: str = 'trained_models'):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.intent_classifier:
            self.intent_classifier.model.save(
                f'{output_dir}/intent_classifier.h5'
            )
        
        if self.building_recognizer:
            with open(f'{output_dir}/building_recognizer.pkl', 'wb') as f:
                pickle.dump(self.building_recognizer, f)