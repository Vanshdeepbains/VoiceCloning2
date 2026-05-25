import sqlite3

conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# Users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# History table
cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    text_input TEXT,
    audio_path TEXT
)
""")

# Credits table
cursor.execute("""
CREATE TABLE IF NOT EXISTS credits(
    username TEXT UNIQUE,
    balance INTEGER DEFAULT 20
)
""")

conn.commit()