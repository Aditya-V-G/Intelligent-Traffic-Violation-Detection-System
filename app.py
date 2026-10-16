from flask import Flask, render_template, request, redirect, session
import os, traceback, uuid
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from werkzeug.security import check_password_hash

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "secret123")
if app.secret_key == "secret123":
    print("⚠ FLASK_SECRET_KEY is not set: using an insecure default. Set it in .env for production.")

BASE_DIR = os.getcwd()
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DETECTIONS_FOLDER = os.path.join(UPLOAD_FOLDER, "detections")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["DETECTIONS_FOLDER"] = DETECTIONS_FOLDER

# The session lives in a browser cookie and processing runs inside the request, so a server
# stopped mid-video (Ctrl+C) never got to remove the old video from the cookie: after a restart
# the browser re-sent it and the OLD video was processed again, and the user was still logged in.
# A new id on every server start makes cookies from before the restart invalid (back to the
# login page), and each upload is a one-time job held in server memory (gone after a restart,
# and a page reload cannot run it twice).
BOOT_ID = uuid.uuid4().hex
PENDING_JOBS = {}        # job id -> (process type, [uploaded file names])
PUBLIC_ENDPOINTS = {"login", "validate_login", "static"}


# ---------------------------------------------------
# SAFE IMPORT LOADER
# ---------------------------------------------------
def safe_import(name, module):
    try:
        m = __import__(module, fromlist=[name])
        return getattr(m, name)
    except Exception as e:
        print(f"Import {name} from {module} failed:", e)
        return None


# ---------------------------------------------------
# IMPORTING MODULES
# ---------------------------------------------------
start_anpr = safe_import("start_anpr", "anpr_video")
process_atcc_videos = safe_import("process_atcc_videos", "atcc")
main_fun = safe_import("main_fun", "helmet_detection")
detect_triple_riding = safe_import("detect_triple_riding", "triple_riding")
detect_accident = safe_import("detect_accident", "accident_detection")
detect_phone_usage = safe_import("detect_phone_usage", "phone_usage")
detect_seatbelt = safe_import("detect_seatbelt", "seatbelt_detection")
detect_wrong_lane = safe_import("detect_wrong_lane", "wrong_lane")
get_connection = safe_import("get_connection", "dbconnection")


# ---------------------------------------------------
# ALLOWED VIDEO FILE TYPES
# ---------------------------------------------------
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ---------------------------------------------------
# LOGIN REQUIRED (and only valid for the current server run)
# ---------------------------------------------------
@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if "user" not in session or session.get("boot") != BOOT_ID:
        session.clear()
        return redirect("/")
    return None


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/validate_login", methods=["POST"])
def validate_login():

    if get_connection is None:
        return render_template("login.html", error="DB helper missing")

    username = request.form.get("username", "").strip()
    password = request.form.get("password", "").strip()

    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM login_details WHERE username=%s", (username,))
            row = cur.fetchone()
        conn.close()

    except Exception as e:
        print("Login DB Error:", e)
        return render_template("login.html", error="DB Error")

    def password_ok(stored):
        stored = stored or ""
        if stored.startswith(("scrypt:", "pbkdf2:")):        # hashed (create_tables.py --add-user)
            return check_password_hash(stored, password)
        return stored == password                             # older rows with plain-text passwords

    if row and password_ok(row.get("password")):
        session.clear()
        session["user"] = username
        session["boot"] = BOOT_ID
        return redirect("/home")
    else:
        return render_template("login.html", error="Invalid login")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


@app.route("/home")
def home():
    return render_template("home.html")


# ---------------------------------------------------
# LIVE MONITORING PAGE
# ---------------------------------------------------
@app.route("/live_monitoring", methods=["GET", "POST"])
def live_monitoring():

    if request.method == "POST":
        num = int(request.form.get("numCameras", 1))
        ptype = request.form.get("processType", "anpr")

        session["num"] = num
        session["processType"] = ptype

        uploaded = []
        for i in range(1, num + 1):
            field = f"cameraFile{i}"
            if field in request.files:
                f = request.files[field]
                if f and f.filename and allowed_file(f.filename):
                    name = secure_filename(f.filename)
                    path = os.path.join(app.config["UPLOAD_FOLDER"], name)
                    f.save(path)
                    uploaded.append(name)

        session["uploaded_files"] = uploaded
        if not uploaded:
            return "No videos uploaded", 400
        job = uuid.uuid4().hex
        PENDING_JOBS[job] = (ptype, uploaded)
        session["job"] = job
        return redirect("/start_processing")

    return render_template(
        "live_monitoring.html",
        numCameras=session.get("num", 1),
        processType=session.get("processType", "anpr"),
        uploaded_files=session.get("uploaded_files", [])
    )


# ---------------------------------------------------
# PROCESSING VIDEO BASED ON TYPE
# ---------------------------------------------------
@app.route("/start_processing")
def start_processing():

    # a job runs once: a reload of this page, or a cookie from before a restart, finds nothing
    job = PENDING_JOBS.pop(session.pop("job", None), None)
    if job is None:
        return redirect("/live_monitoring")
    ptype, uploaded = job

    input_paths = [os.path.join(app.config["UPLOAD_FOLDER"], f) for f in uploaded]

    try:
        # PHONE USAGE
        if ptype == "phone":
            for v in input_paths:
                detect_phone_usage(v, app.config["DETECTIONS_FOLDER"])

        # SEATBELT
        elif ptype == "seatbelt":
            for v in input_paths:
                detect_seatbelt(v, app.config["DETECTIONS_FOLDER"])

        # WRONG LANE
        elif ptype == "wronglane":
            from wrong_lane import detect_wrong_lane
            for v in input_paths:
                detect_wrong_lane(v, app.config["DETECTIONS_FOLDER"])

        # ACCIDENT
        elif ptype == "accident":
            for v in input_paths:
                detect_accident(v, app.config["DETECTIONS_FOLDER"])

        # TRIPLE RIDING
        elif ptype == "triple":
            for v in input_paths:
                detect_triple_riding(v, app.config["DETECTIONS_FOLDER"])

        # HELMET
        elif ptype == "helmet":
            for v in input_paths:
                main_fun(v, app.config["DETECTIONS_FOLDER"])

        # ATCC
        elif ptype == "atcc":
            from ultralytics import YOLO
            model = YOLO("trainedmodels/vehicledetection.pt")
            process_atcc_videos(input_paths, model, app.config["DETECTIONS_FOLDER"])

        # ANPR
        elif ptype == "anpr":
            start_anpr(input_paths, app.config["DETECTIONS_FOLDER"])

    except Exception as e:
        print("Processing Error:", e)
        traceback.print_exc()
        return f"Processing error: {e}", 500

    session.pop("uploaded_files", None)
    return redirect("/results")


# ---------------------------------------------------
# RESULTS PAGE
# ---------------------------------------------------
@app.route("/results")
def results():

    rows = []
    if get_connection:
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM vehicle_data ORDER BY detected_at DESC")
                rows = cur.fetchall()
            conn.close()
        except:
            rows = []

    grouped = {
        "helmet_violation": [],
        "triple_riding": [],
        "phone_usage": [],
        "seatbelt_violation": [],
        "wrong_lane": [],
        "accident": [],
        "atcc": [],
        "others": []
    }

    for row in rows:
        module = row.get("module", "")
        if module in grouped:
            grouped[module].append(row)
        else:
            grouped["others"].append(row)

    return render_template("results.html", grouped=grouped)

@app.route("/clear_database")
def clear_database():
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE vehicle_data")
            cur.execute("TRUNCATE TABLE license_plates")
        conn.close()

        print("✔ Database cleared successfully.")

    except Exception as e:
        print("❌ Error clearing DB:", e)
        return "Database clear failed!", 500

    return redirect("/results")

# ---------------------------------------------------
# RUN SERVER
# ---------------------------------------------------
if __name__ == "__main__":
    print("UPLOAD:", app.config["UPLOAD_FOLDER"])
    print("DETECTIONS:", app.config["DETECTIONS_FOLDER"])
    # debug mode only when asked for (it allows running code from the browser on errors)
    app.run(host=os.getenv("FLASK_HOST", "127.0.0.1"), port=int(os.getenv("FLASK_PORT", 5000)),
            debug=os.getenv("FLASK_DEBUG", "0") == "1")
