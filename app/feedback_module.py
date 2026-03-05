# feedback_module.py
from flask import Blueprint, request, jsonify
import sqlite3
from datetime import datetime
import json

feedback_bp = Blueprint('feedback', __name__)

# Initialize database
def init_feedback_db():
    """Initialize feedback database"""
    conn = sqlite3.connect('ai_feedback.db')
    cursor = conn.cursor()
    
    # Feedback table
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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT,
            user_agent TEXT
        )
    ''')
    
    # User queries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            language TEXT,
            intent_detected TEXT,
            building_detected TEXT,
            confidence_intent REAL,
            confidence_building REAL,
            success BOOLEAN,
            response_time REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT
        )
    ''')
    
    # Query patterns table (for learning new patterns)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS query_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            intent TEXT,
            building TEXT,
            frequency INTEGER DEFAULT 1,
            last_used DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Feedback database initialized")

# Initialize database on import
init_feedback_db()

@feedback_bp.route('/log_query', methods=['POST'])
def log_query():
    """Log user query for training"""
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'No query provided'}), 400
        
        conn = sqlite3.connect('ai_feedback.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO user_queries 
            (query, language, intent_detected, building_detected, 
             confidence_intent, confidence_building, success, 
             response_time, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('query'),
            data.get('language', 'english'),
            data.get('intent'),
            data.get('building'),
            data.get('intent_confidence', 0.0),
            data.get('building_confidence', 0.0),
            data.get('success', False),
            data.get('processing_time', 0.0),
            request.remote_addr
        ))
        
        # Also store successful patterns for learning
        if data.get('success') and data.get('intent') and data.get('building'):
            cursor.execute('''
                INSERT OR REPLACE INTO query_patterns 
                (pattern, intent, building, frequency, last_used)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT frequency FROM query_patterns 
                             WHERE pattern = ? AND intent = ? AND building = ?), 0) + 1,
                    CURRENT_TIMESTAMP)
            ''', (
                data.get('query'),
                data.get('intent'),
                data.get('building'),
                data.get('query'),
                data.get('intent'),
                data.get('building')
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Query logged successfully'})
    
    except Exception as e:
        print(f"Error logging query: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@feedback_bp.route('/submit_correction', methods=['POST'])
def submit_correction():
    """Submit correction for AI mistakes"""
    try:
        data = request.json
        required_fields = ['original_query', 'predicted_intent', 'predicted_building']
        
        if not data or not all(field in data for field in required_fields):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        conn = sqlite3.connect('ai_feedback.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback 
            (original_query, predicted_intent, predicted_building, 
             correct_intent, correct_building, user_correction, 
             confidence_score, ip_address, user_agent)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('original_query'),
            data.get('predicted_intent'),
            data.get('predicted_building'),
            data.get('correct_intent'),
            data.get('correct_building'),
            data.get('user_correction', ''),
            data.get('confidence_score', 0.0),
            request.remote_addr,
            request.user_agent.string if request.user_agent else ''
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True, 
            'message': 'Correction submitted. Thank you!'
        })
    
    except Exception as e:
        print(f"Error submitting correction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@feedback_bp.route('/get_stats', methods=['GET'])
def get_stats():
    """Get feedback statistics"""
    try:
        conn = sqlite3.connect('ai_feedback.db')
        cursor = conn.cursor()
        
        # Get total queries
        cursor.execute('SELECT COUNT(*) FROM user_queries')
        total_queries = cursor.fetchone()[0]
        
        # Get successful queries
        cursor.execute('SELECT COUNT(*) FROM user_queries WHERE success = 1')
        successful_queries = cursor.fetchone()[0]
        
        # Get corrections count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        corrections = cursor.fetchone()[0]
        
        # Get accuracy rate
        accuracy = (successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        # Get recent queries
        cursor.execute('''
            SELECT query, intent_detected, building_detected, success, timestamp
            FROM user_queries 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''')
        recent_queries = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            'success': True,
            'stats': {
                'total_queries': total_queries,
                'successful_queries': successful_queries,
                'corrections': corrections,
                'accuracy_rate': round(accuracy, 2),
                'recent_queries': [
                    {
                        'query': q[0],
                        'intent': q[1],
                        'building': q[2],
                        'success': bool(q[3]),
                        'timestamp': q[4]
                    }
                    for q in recent_queries
                ]
            }
        })
    
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@feedback_bp.route('/export_data', methods=['GET'])
def export_data():
    """Export training data for model retraining"""
    try:
        format_type = request.args.get('format', 'json')
        limit = request.args.get('limit', 1000, type=int)
        
        conn = sqlite3.connect('ai_feedback.db')
        cursor = conn.cursor()
        
        # Get successful queries for training
        cursor.execute('''
            SELECT query, intent_detected, building_detected, language
            FROM user_queries 
            WHERE success = 1
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        training_data = cursor.fetchall()
        
        # Get corrections for training
        cursor.execute('''
            SELECT original_query, correct_intent, correct_building
            FROM feedback
            WHERE correct_intent IS NOT NULL AND correct_building IS NOT NULL
            LIMIT ?
        ''', (limit // 2,))
        
        corrections_data = cursor.fetchall()
        
        conn.close()
        
        # Combine data
        combined_data = []
        
        for item in training_data:
            combined_data.append({
                'query': item[0],
                'intent': item[1],
                'building': item[2],
                'language': item[3] or 'english',
                'source': 'user_query'
            })
        
        for item in corrections_data:
            combined_data.append({
                'query': item[0],
                'intent': item[1],
                'building': item[2],
                'language': 'english',  # Default
                'source': 'correction'
            })
        
        if format_type == 'json':
            return jsonify({
                'success': True,
                'count': len(combined_data),
                'data': combined_data
            })
        
        elif format_type == 'csv':
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.DictWriter(output, 
                                   fieldnames=['query', 'intent', 'building', 'language', 'source'])
            writer.writeheader()
            writer.writerows(combined_data)
            
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=training_data.csv'
            }
        
        else:
            return jsonify({'success': False, 'error': 'Invalid format'}), 400
    
    except Exception as e:
        print(f"Error exporting data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500