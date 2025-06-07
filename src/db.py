import sqlite3

def init_db():
    conn = sqlite3.connect('chat_logs.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT,
            ai_response TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_interaction(user_query, ai_response):
    conn = sqlite3.connect('chat_logs.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_logs (user_query, ai_response)
        VALUES (?, ?)
    ''', (user_query, ai_response))
    conn.commit()
    conn.close()

# Call init_db() once at app startup
init_db()
