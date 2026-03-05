# train_models.py
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, models_dir='ai_models'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.intent_vectorizer = TfidfVectorizer(max_features=1000)
        self.building_vectorizer = TfidfVectorizer(max_features=1000)
        self.intent_classifier = None
        self.building_classifier = None
    
    def load_data(self, data_files):
        """Load training data from multiple sources"""
        all_data = []
        
        for data_file in data_files:
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    if data_file.endswith('.json'):
                        data = json.load(f)
                        all_data.extend(data)
                    elif data_file.endswith('.csv'):
                        df = pd.read_csv(data_file)
                        all_data.extend(df.to_dict('records'))
        
        if not all_data:
            raise ValueError("No training data found")
        
        return pd.DataFrame(all_data)
    
    def preprocess_data(self, df):
        """Preprocess training data"""
        # Clean queries
        df['clean_query'] = df['query'].str.lower().str.strip()
        
        # Handle missing values
        df['intent'] = df['intent'].fillna('unknown')
        df['building'] = df['building'].fillna('')
        
        return df
    
    def train_intent_classifier(self, X_train, y_train):
        """Train intent classification model"""
        print("🔄 Training intent classifier...")
        
        # Vectorize text
        X_train_vec = self.intent_vectorizer.fit_transform(X_train)
        
        # Train classifier
        self.intent_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.intent_classifier.fit(X_train_vec, y_train)
        
        print(f"✅ Intent classifier trained on {len(X_train)} samples")
    
    def train_building_classifier(self, X_train, y_train):
        """Train building recognition model"""
        print("🔄 Training building recognizer...")
        
        # Only train on queries with buildings
        mask = y_train != ''
        if mask.sum() == 0:
            print("⚠ No building data to train on")
            return
        
        X_train_buildings = X_train[mask]
        y_train_buildings = y_train[mask]
        
        # Vectorize text
        X_train_vec = self.building_vectorizer.fit_transform(X_train_buildings)
        
        # Train classifier
        self.building_classifier = RandomForestClassifier(
            n_estimators=150,
            random_state=42
        )
        self.building_classifier.fit(X_train_vec, y_train_buildings)
        
        print(f"✅ Building recognizer trained on {len(X_train_buildings)} samples")
    
    def evaluate_models(self, X_test, y_test_intent, y_test_building):
        """Evaluate model performance"""
        results = {}
        
        # Evaluate intent classifier
        if self.intent_classifier:
            X_test_vec = self.intent_vectorizer.transform(X_test)
            accuracy = self.intent_classifier.score(X_test_vec, y_test_intent)
            results['intent_accuracy'] = accuracy
        
        # Evaluate building classifier
        if self.building_classifier:
            mask = y_test_building != ''
            if mask.sum() > 0:
                X_test_buildings = X_test[mask]
                y_test_buildings = y_test_building[mask]
                X_test_vec = self.building_vectorizer.transform(X_test_buildings)
                accuracy = self.building_classifier.score(X_test_vec, y_test_buildings)
                results['building_accuracy'] = accuracy
        
        return results
    
    def save_models(self):
        """Save trained models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save intent model
        if self.intent_classifier:
            joblib.dump(self.intent_classifier, 
                       f'{self.models_dir}/intent_classifier_{timestamp}.pkl')
            joblib.dump(self.intent_vectorizer,
                       f'{self.models_dir}/intent_vectorizer_{timestamp}.pkl')
        
        # Save building model
        if self.building_classifier:
            joblib.dump(self.building_classifier,
                       f'{self.models_dir}/building_classifier_{timestamp}.pkl')
            joblib.dump(self.building_vectorizer,
                       f'{self.models_dir}/building_vectorizer_{timestamp}.pkl')
        
        # Create symlink to latest models
        latest_dir = f'{self.models_dir}/latest'
        os.makedirs(latest_dir, exist_ok=True)
        
        if self.intent_classifier:
            joblib.dump(self.intent_classifier, f'{latest_dir}/intent_classifier.pkl')
            joblib.dump(self.intent_vectorizer, f'{latest_dir}/intent_vectorizer.pkl')
        
        if self.building_classifier:
            joblib.dump(self.building_classifier, f'{latest_dir}/building_classifier.pkl')
            joblib.dump(self.building_vectorizer, f'{latest_dir}/building_vectorizer.pkl')
        
        print(f"💾 Models saved to {self.models_dir}")
    
    def train(self, data_files, test_size=0.2):
        """Complete training pipeline"""
        print("🚀 Starting model training...")
        
        # Load data
        df = self.load_data(data_files)
        print(f"📊 Loaded {len(df)} training examples")
        
        # Preprocess
        df = self.preprocess_data(df)
        
        # Split data
        X = df['clean_query'].values
        y_intent = df['intent'].values
        y_building = df['building'].values
        
        X_train, X_test, y_train_intent, y_test_intent = train_test_split(
            X, y_intent, test_size=test_size, random_state=42, stratify=y_intent
        )
        
        _, _, y_train_building, y_test_building = train_test_split(
            X, y_building, test_size=test_size, random_state=42
        )
        
        # Train models
        self.train_intent_classifier(X_train, y_train_intent)
        self.train_building_classifier(X_train, y_train_building)
        
        # Evaluate
        results = self.evaluate_models(X_test, y_test_intent, y_test_building)
        
        print("\n📈 Training Results:")
        for metric, value in results.items():
            print(f"   {metric}: {value:.2%}")
        
        # Save models
        self.save_models()
        
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI models for campus navigation')
    parser.add_argument('--data', nargs='+', default=['training_data.json'],
                       help='Training data files (JSON/CSV)')
    parser.add_argument('--models-dir', default='ai_models',
                       help='Directory to save models')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size proportion')
    
    args = parser.parse_args()
    
    trainer = ModelTrainer(models_dir=args.models_dir)
    results = trainer.train(args.data, test_size=args.test_size)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Models saved in: {args.models_dir}")

if __name__ == '__main__':
    main()