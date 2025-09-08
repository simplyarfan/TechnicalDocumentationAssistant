import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

class SessionDatabase:
    def __init__(self, db_path: str = "sessions.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                documents_processed INTEGER,
                total_questions INTEGER,
                processing_stats TEXT
            )
        ''')
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                answer TEXT,
                timestamp TIMESTAMP,
                sources_count INTEGER,
                search_type TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session(self, session_data: Dict[str, Any]):
        """Save session metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO sessions 
            (session_id, start_time, end_time, documents_processed, total_questions, processing_stats)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_data['session_id'],
            session_data['start_time'],
            session_data.get('end_time'),
            session_data.get('documents_processed', 0),
            session_data.get('total_questions', 0),
            json.dumps(session_data.get('processing_stats', {}))
        ))
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, conversation_item: Dict[str, Any]):
        """Save individual conversation item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, question, answer, timestamp, sources_count, search_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            session_id,
            conversation_item['question'],
            conversation_item['answer'],
            conversation_item['timestamp'],
            conversation_item['sources_count'],
            conversation_item['search_type']
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_analytics(self) -> Dict[str, Any]:
        """Get analytics across all sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Session stats
        cursor.execute('SELECT COUNT(*) FROM sessions')
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(total_questions) FROM sessions')
        total_questions = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT AVG(total_questions) FROM sessions WHERE total_questions > 0')
        avg_questions = cursor.fetchone()[0] or 0
        
        # Most common question patterns
        cursor.execute('''
            SELECT question, COUNT(*) as count 
            FROM conversations 
            GROUP BY LOWER(SUBSTR(question, 1, 20))
            ORDER BY count DESC 
            LIMIT 5
        ''')
        common_patterns = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_questions": total_questions,
            "avg_questions_per_session": round(avg_questions, 2),
            "common_question_patterns": common_patterns
        }