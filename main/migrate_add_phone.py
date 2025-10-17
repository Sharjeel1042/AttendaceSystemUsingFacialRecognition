import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'attendance.db')

if not os.path.exists(DB_PATH):
    print(f"[ERROR] Database not found at {DB_PATH}")
    exit(1)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Check if phone_number column exists
cur.execute("PRAGMA table_info(students);")
cols = cur.fetchall()
col_names = [c[1] for c in cols]

if 'phone_number' in col_names:
    print('[INFO] phone_number column already exists in students table.')
else:
    try:
        cur.execute("ALTER TABLE students ADD COLUMN phone_number TEXT DEFAULT ''")
        conn.commit()
        print('[INFO] Added phone_number column to students table.')
    except sqlite3.OperationalError as e:
        print(f'[ERROR] Could not add column: {e}')

conn.close()

