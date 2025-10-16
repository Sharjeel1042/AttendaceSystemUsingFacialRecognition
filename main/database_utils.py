import sqlite3
from datetime import datetime
import os

DB_FOLDER = "../database"
DB_PATH = os.path.join(DB_FOLDER, "attendance.db")


def init_database():
    """
    Initializes the SQLite database and creates the necessary tables.
    """
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Table to store student details
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            reg_no TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            semester TEXT,
            phone_number TEXT
        )
    """)

    # Table to log attendance records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_reg_no TEXT,
            timestamp TEXT,
            FOREIGN KEY (student_reg_no) REFERENCES students (reg_no)
        )
    """)
    conn.commit()
    print("[INFO] Database initialized successfully.")
    return conn


def add_student(conn, reg_no, name, semester, phone):
    """Adds a new student to the students table if they don't already exist."""
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO students (reg_no, name, semester, phone_number) VALUES (?, ?, ?, ?)",
                       (reg_no, name, semester, phone))
        conn.commit()
        print(f"[INFO] Student {name} ({reg_no}) added to the database.")
    except sqlite3.IntegrityError:
        # This error occurs if the reg_no (PRIMARY KEY) already exists.
        # It's a safe way to avoid duplicate entries.
        pass


def mark_attendance(conn, reg_no):
    """
    Marks attendance for a recognized student.
    It checks if attendance has already been marked for that person today to avoid duplicates.
    """
    if not reg_no:
        return
    cursor = conn.cursor()
    today_date = datetime.now().strftime("%Y-%m-%d")

    # Check if an entry for this student on this day already exists
    cursor.execute("SELECT * FROM attendance WHERE student_reg_no = ? AND date(timestamp) = ?", (reg_no, today_date))
    if cursor.fetchone():
        return  # Already marked today, so do nothing

    # If not marked, insert a new record
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO attendance (student_reg_no, timestamp) VALUES (?, ?)", (reg_no, now))
    conn.commit()

    # Get student name for a more descriptive log message
    cursor.execute("SELECT name FROM students WHERE reg_no = ?", (reg_no,))
    result = cursor.fetchone()
    name = result[0] if result else "Unknown"

    print(f"[ATTENDANCE] Marked for {name} ({reg_no}) at {now}")

