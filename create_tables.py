# create_tables.py
"""
Create the database tables (safe to run again).
    python create_tables.py                      # tables only
    python create_tables.py --add-user admin     # also create / reset a login (asks for the password)
"""
import argparse
import getpass

from werkzeug.security import generate_password_hash

from dbconnection import get_connection

def create():
    conn = get_connection()
    cur = conn.cursor()
    # vehicle_data: main results table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        number_plate VARCHAR(50),
        video_file VARCHAR(255),
        module VARCHAR(50),
        image_filename VARCHAR(500),
        extra_info TEXT,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_plate (number_plate),
        INDEX idx_module (module)
    )
    """)
    # columns match dbconnection.insert_detection / update_license_plate
    # license_plates: one row per plate with its violation count
    cur.execute("""
    CREATE TABLE IF NOT EXISTS license_plates (
        id INT AUTO_INCREMENT PRIMARY KEY,
        plate_number VARCHAR(50) UNIQUE,
        violation_count INT DEFAULT 1,
        last_seen DATETIME
    )
    """)
    # login_details: web-app users (app.py logs in against this table). Passwords are stored hashed;
    # rows that still hold a plain-text password keep working.
    cur.execute("""
    CREATE TABLE IF NOT EXISTS login_details (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(255) NOT NULL
    )
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Tables created/checked.")


def add_user(username, password):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("INSERT INTO login_details (username, password) VALUES (%s, %s) "
                    "ON DUPLICATE KEY UPDATE password = VALUES(password)",
                    (username, generate_password_hash(password)))
    conn.commit()
    conn.close()
    print(f"✅ User '{username}' saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add-user", metavar="USERNAME")
    ap.add_argument("--password", help="omit to be asked (keeps it out of the shell history)")
    args = ap.parse_args()
    create()
    if args.add_user:
        add_user(args.add_user, args.password or getpass.getpass(f"Password for {args.add_user}: "))
