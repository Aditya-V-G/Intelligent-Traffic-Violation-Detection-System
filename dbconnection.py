import os
import pymysql
from dotenv import load_dotenv
import traceback

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "mydb")
DB_PORT = int(os.getenv("DB_PORT", 3306))

def get_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=DB_PORT,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True
        )
        return conn
    except Exception as e:
        print("❌ DB CONNECTION FAILED:", e)
        raise

def insert_detection(*args, **kwargs):
    if len(args) >= 4:
        number_plate, video_file, module, image_filename = args[:4]
        extra_info = args[4] if len(args) > 4 else kwargs.get("extra_info", "")
    else:
        number_plate = kwargs.get("number_plate") or kwargs.get("plate") or kwargs.get("numberPlate") or ""
        video_file = kwargs.get("video_file") or kwargs.get("videoFile") or ""
        module = kwargs.get("module") or kwargs.get("process_type") or kwargs.get("processType") or ""
        image_filename = kwargs.get("image_filename") or kwargs.get("image_filename") or ""
        extra_info = kwargs.get("extra_info", "") or kwargs.get("extraInfo", "")
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO vehicle_data (number_plate, video_file, module, image_filename, extra_info) VALUES (%s,%s,%s,%s,%s)",
                (number_plate, video_file, module, image_filename, extra_info)
            )
        conn.close()
    except Exception as e:
        print("❌ insert_detection ERROR:", e)
        print(traceback.format_exc())

def update_license_plate(plate):
    if not plate:
        return
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM license_plates WHERE plate_number=%s", (plate,))
            row = cur.fetchone()
            if row:
                cur.execute("UPDATE license_plates SET violation_count = violation_count + 1, last_seen = NOW() WHERE plate_number=%s", (plate,))
            else:
                cur.execute("INSERT INTO license_plates (plate_number, violation_count, last_seen) VALUES (%s, 1, NOW())", (plate,))
        conn.close()
    except Exception as e:
        print("❌ update_license_plate ERROR:", e)
        print(traceback.format_exc())
