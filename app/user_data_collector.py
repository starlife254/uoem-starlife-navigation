# user_data_collector.py
from flask import Blueprint, request, jsonify
from datetime import datetime
import sqlite3

feedback_bp = Blueprint('feedback', __name__)

class FeedbackCollector:
    def __init__(self, db_path='ai_feedback.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize feedback database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_query TEXT NOT NULL,
                predicted_intent TEXT,
                predicted_building TEXT,
                correct_intent TEXT,
                correct_building TEXT,
                user_correction TEXT,
                confidence_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                language TEXT,
                intent_detected TEXT,
                building_detected TEXT,
                success BOOLEAN,
                response_time REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_query(self, query_data: Dict):
        """Log all user queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_queries 
            (query, language, intent_detected, building_detected, success, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            query_data.get('query'),
            query_data.get('language'),
            query_data.get('intent'),
            query_data.get('building'),
            query_data.get('success'),
            query_data.get('processing_time')
        ))
        
        conn.commit()
        conn.close()
    
    def log_feedback(self, feedback_data: Dict):
        """Log user corrections"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (original_query, predicted_intent, predicted_building, 
             correct_intent, correct_building, user_correction, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_data.get('original_query'),
            feedback_data.get('predicted_intent'),
            feedback_data.get('predicted_building'),
            feedback_data.get('correct_intent'),
            feedback_data.get('correct_building'),
            feedback_data.get('user_correction'),
            feedback_data.get('confidence_score')
        ))
        
        conn.commit()
        conn.close()