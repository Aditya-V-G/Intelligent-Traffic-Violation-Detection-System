from dbconnection import get_connection

try:
    conn = get_connection()
    print("DB OK")
except Exception as e:
    print("DB FAIL:", e)
