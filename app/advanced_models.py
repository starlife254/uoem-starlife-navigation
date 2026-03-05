# advanced_models.py
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import BertTokenizer, TFBertModel
import numpy as np

class AdvancedIntentClassifier:
    def __init__(self, num_intents: int, model_type: str = 'bert'):
        self.model_type = model_type
        self.num_intents = num_intents
        
        if model_type == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = TFBertModel.from_pretrained('bert-base-uncased')
            self.model = self.build_bert_model()
        else:
            self.model = self.build_cnn_model()
    
    def build_bert_model(self):
        """Build BERT-based intent classifier"""
        input_ids = layers.Input(shape=(128,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(128,), dtype=tf.int32)
        
        bert_output = self.bert_model(input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        
        x = layers.Dropout(0.3)(pooled_output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(128, activation='relu')(x)
        outputs = layers.Dense(self.num_intents, activation='softmax')(x)
        
        model = models.Model(
            inputs=[input_ids, attention_mask], 
            outputs=outputs
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self):
        """Build CNN-based intent classifier"""
        model = models.Sequential([
            layers.Embedding(10000, 128, input_length=100),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_intents, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_bert(self, texts: List[str]):
        """Preprocess text for BERT"""
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='tf'
        )
        
        return encoding['input_ids'], encoding['attention_mask']