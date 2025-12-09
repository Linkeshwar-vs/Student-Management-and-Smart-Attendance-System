from flask import Blueprint, render_template, redirect, url_for, session, flash, request
from flask import Blueprint, render_template, request, jsonify, session, flash, redirect, url_for
from .models import Student, Class, Attendance, AttendanceRecord,Quiz , QuizQuestion, QuizOption, QuizAttempt , QuizAnswer
from . import db
from datetime import timedelta, datetime
import os
from flask_login import login_required, current_user
import random, smtplib
from .utils import student_login_required
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


###################### SPOOF DETECTION ########################
import torch
import cv2
import numpy as np
from torchvision import transforms
from model import ResNet_CDCNN   # make sure model.py is in the same folder
from PIL import Image
# Define preprocessing transform (same as training)
# transforms (no ToPILImage here)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# Load model once (keep global for Flask)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet_CDCNN(num_classes=2).to(device)
model.load_state_dict(torch.load("resnet_cdcnn.pth", map_location=device))
model.eval()

def predict_image(img_np):
    """
    Takes OpenCV image (numpy array BGR), preprocesses and predicts.
    Returns: label (str), confidence (float)
    """
    # Ensure BGR -> RGB
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image explicitly
    img_pil = Image.fromarray(img_rgb.astype("uint8"))

    # Apply transform
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    idx_to_class = {0: "real", 1: "spoof"}  # check your dataset order
    return idx_to_class[pred.item()], conf.item()

###############################################################
# Blueprint for student-related pages
student_bp = Blueprint("student", __name__)

@student_bp.route("/student/dashboard")
def dashboard():
    student_id = session.get("student_id")
    if not student_id:
        flash("Please log in first", "warning")
        return redirect(url_for("student_auth.login"))

    student = Student.query.get(student_id)
    if not student:
        flash("Student not found.", "danger")
        return redirect(url_for("student_auth.login"))

    # 🕒 Deactivate old attendance sessions (>1 hr)
    ongoing_sessions = Attendance.query.filter_by(is_active=True).all()
    for session_att in ongoing_sessions:
        if datetime.utcnow() - session_att.created_at > timedelta(hours=1):
            session_att.is_active = False
    db.session.commit()

    my_classes = student.classes
    class_ids = [cls.id for cls in my_classes]

    ongoing = Attendance.query.filter(
        Attendance.is_active == True,
        Attendance.class_id.in_(class_ids)
    ).order_by(Attendance.created_at.desc()).all()

    quizzes = Quiz.query.filter(
        Quiz.class_id.in_(class_ids),
        Quiz.is_active == True
    ).all()

    # Use IST timezone
    import pytz
    IST = pytz.timezone("Asia/Kolkata")
    now = datetime.now(IST)

    print(f"=== DASHBOARD DEBUG @ {now} ===")
    completed_ids = [
        a.quiz_id for a in QuizAttempt.query.filter_by(student_id=student_id)
        if a.submitted_at is not None
    ]

    live_quizzes, locked_quizzes, closed_quizzes = [], [], []

    for q in quizzes:
        if q.id in completed_ids:
            continue

        start = q.start_time.astimezone(IST) if q.start_time else None
        end = q.end_time.astimezone(IST) if q.end_time else None
        print(f"Quiz {q.id}: start={start}, end={end}, active={q.is_active}")

        if start and now < start:
            locked_quizzes.append(q)
        elif end and now > end:
            closed_quizzes.append(q)
        else:
            live_quizzes.append(q)

    return render_template(
        "student_dashboard.html",
        student=student,
        classes=my_classes,
        ongoing=ongoing,
        live_quizzes=live_quizzes,
        locked_quizzes=locked_quizzes,
        closed_quizzes=closed_quizzes,
        current_time=now
    )





# -------------------- View Attendance --------------------
@student_bp.route("/student/attendance/<int:class_id>")
def view_attendance(class_id):
    student_id = session.get("student_id")
    if not student_id:
        flash("Please log in first", "warning")
        return redirect(url_for("student_auth.login"))


    # Check if student belongs to the class
    student = Student.query.get(student_id)
    if not student or not any(c.id == class_id for c in student.classes):
        flash("You are not enrolled in this class", "danger")
        return redirect(url_for("student.dashboard"))

    # Fetch attendance records
    
    attendance_records = (
        AttendanceRecord.query
        .join(Attendance)
        .filter(
            Attendance.class_id == class_id,
            AttendanceRecord.student_id == student_id
        )
        .all()
    )

    return render_template("student_attendance.html", student=student, attendance_records=attendance_records)


# -------------------- Join Class (Optional) --------------------
@student_bp.route("/student/join_class", methods=["POST"])
def join_class():
    student_id = session.get("student_id")
    if not student_id:
        flash("Please log in first", "warning")
        return redirect(url_for("student_auth.login"))


    class_code = request.form.get("class_code")
    target_class = Class.query.filter_by(code=class_code).first()

    if not target_class:
        flash("Invalid class code", "danger")
        return redirect(url_for("student.dashboard"))

    student = Student.query.get(student_id)
    if target_class in student.classes:
        flash("You are already enrolled in this class", "info")
    else:
        student.classes.append(target_class)
        db.session.commit()
        flash(f"Successfully joined class {target_class.name}", "success")

    return redirect(url_for("student.dashboard"))

# @student_bp.route("/student/attendance/mark/<int:attendance_id>", methods=["POST"])
# def mark_attendance(attendance_id):
#     student_id = session.get("student_id")
#     if not student_id:
#         flash("Please log in first", "warning")
#         return redirect(url_for("student_auth.student_login"))

#     # Check if attendance session exists
#     attendance = Attendance.query.get(attendance_id)
#     if not attendance or not attendance.is_active:
#         flash("Attendance session not found or inactive.", "danger")
#         return redirect(url_for("student.dashboard"))

#     # Check if student already marked
#     existing_record = AttendanceRecord.query.filter_by(
#         attendance_id=attendance_id,
#         student_id=student_id
#     ).first()

#     if existing_record:
#         flash("You have already marked this attendance.", "info")
#     else:
#         # Mark present
#         new_record = AttendanceRecord(
#             attendance_id=attendance_id,
#             student_id=student_id,
#             is_present=True,
#             timestamp=datetime.utcnow()
#         )
#         db.session.add(new_record)
#         db.session.commit()
#         flash("Attendance marked successfully.", "success")

#     return redirect(url_for("student.dashboard"))
from math import radians, cos, sin, asin, sqrt

@student_bp.route("/student/attendance/mark/<int:attendance_id>", methods=["GET"])
def mark_attendance_page(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        flash("Please log in first", "warning")
        return redirect(url_for("student_auth.login"))

    attendance = Attendance.query.get(attendance_id)
    if not attendance or not attendance.is_active:
        flash("Attendance session not found or inactive.", "danger")
        return redirect(url_for("student.dashboard"))

    methods = attendance.methods.split(",") if attendance.methods else []

    return render_template(
        "mark_attendance.html",
        attendance=attendance,
        methods=methods
    )

# -------------------- Mark Attendance API (POST) --------------------
@student_bp.route("/student/attendance/mark/<int:attendance_id>", methods=["POST"])
def mark_attendance_api(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    student = Student.query.get(student_id)
    attendance = Attendance.query.get(attendance_id)

    if not attendance or not attendance.is_active:
        return jsonify({"success": False, "error": "Attendance session not found or inactive."}), 404

    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON data"}), 400

    # Track which methods are enabled
    methods = attendance.methods.split(",") if attendance.methods else []
    validation_results = {}

    # Manual attendance
    if "manual" in methods and data.get("manual"):
        validation_results["manual"] = True

    # QR
    if "qr" in methods:
        validation_results["qr"] = data.get("qr_code") == attendance.current_qr

    # Face
    if "face" in methods:
        validation_results["face"] = data.get("face_verified", False)

    # Geo
    if "geo" in methods:
        user_lat = float(data.get("lat", 0))
        user_lng = float(data.get("lng", 0))

        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371000
            return c * r

        distance = haversine(user_lat, user_lng, attendance.geo_lat, attendance.geo_lng)
        validation_results["geo"] = distance <= attendance.geo_radius

    # Mark attendance if all validations pass
    if all(validation_results.values()):
        existing_record = AttendanceRecord.query.filter_by(
            attendance_id=attendance_id, student_id=student_id
        ).first()
        if not existing_record:
            record = AttendanceRecord(
                attendance_id=attendance_id,
                student_id=student_id,
                is_present=True,
                timestamp=datetime.utcnow()
            )
            db.session.add(record)
            db.session.commit()
        return jsonify({"success": True, "message": "Attendance marked!"})
    else:
        return jsonify({"success": False, "validation": validation_results})

import re
from urllib.parse import urlparse, parse_qs
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
from flask import jsonify, request, session
from .models import Attendance, AttendanceRecord, Student
from . import db

# ---------------- helpers ----------------

def _required_methods(att: Attendance):
    """
    Parse Attendance.methods and drop 'manual'. Only return supported keys.
    """
    if not att.methods:
        return []
    allowed = {"qr", "geo", "face"}  # manual ignored
    return [m for m in (x.strip() for x in att.methods.split(",")) if m in allowed]

def _extract_qr_code(scanned: str):
    """
    Accepts:
      - plain 8-digit code: '12345678'
      - absolute/relative URL like '/attendance/74/validate/12345678'
      - URL with ?code=12345678
      - any string containing an 8-digit block
    Returns the 8-digit string or None.
    """
    if not scanned:
        return None

    # If it's exactly 8 digits
    if re.fullmatch(r"\d{8}", scanned):
        return scanned

    # Try parse as URL
    try:
        p = urlparse(scanned)
        # path last segment
        if p.path:
            last = p.path.rstrip("/").split("/")[-1]
            if re.fullmatch(r"\d{8}", last):
                return last
        # query param ?code=XXXXXXXX
        qs_code = parse_qs(p.query).get("code", [None])[0]
        if qs_code and re.fullmatch(r"\d{8}", qs_code):
            return qs_code
    except Exception:
        pass

    # Fallback: first 8 consecutive digits in the string
    m = re.search(r"\b(\d{8})\b", scanned)
    return m.group(1) if m else None

def _haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000  # meters
    return c * r

def _get_or_create_record(student_id: int, attendance_id: int) -> AttendanceRecord:
    rec = AttendanceRecord.query.filter_by(
        student_id=student_id, attendance_id=attendance_id
    ).first()
    if not rec:
        rec = AttendanceRecord(student_id=student_id, attendance_id=attendance_id)
        db.session.add(rec)
        db.session.commit()
    return rec
@student_bp.route("/attendance/status/<int:attendance_id>")
def attendance_status(attendance_id):
    att = Attendance.query.get_or_404(attendance_id)
    return jsonify({"ended": att.has_ended})  # you decide how "ended" is stored



@student_bp.route("/student/submit_qr/<int:attendance_id>", methods=["POST"])
def submit_qr(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    att = Attendance.query.get_or_404(attendance_id)

    # 🚨 Block if session is closed
    if not att.is_active:
        return jsonify({"success": False, "error": "Session ended"}), 400

    # Get posted QR code
    data = request.get_json(silent=True) or {}
    scanned = data.get("qr")
    code = _extract_qr_code(scanned)

    if not code:
        return jsonify({"success": False, "error": "No valid QR found"}), 400

    # Validate against teacher's QR
    if code != att.current_qr:
        return jsonify({"success": False, "error": "Invalid QR"}), 400

    # Fetch or create record
    record = _get_or_create_record(student_id, att.id)
    record.qr_valid = True

    # ✅ Only finalize if session still active & requirements met
    required = _required_methods(att)
    record.update_status(required)
    db.session.commit()

    return jsonify({
        "success": True,
        "present": record.is_present,
        "validated": {
            "qr": record.qr_valid,
            "geo": record.geo_valid,
            "face": record.face_valid,
        },
        "required_methods": required
    })


@student_bp.route("/student/submit_geo/<int:attendance_id>", methods=["POST"])
def submit_geo(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    att = Attendance.query.get_or_404(attendance_id)

    # 🚨 Block if session closed
    if not att.is_active:
        return jsonify({"success": False, "error": "Session ended"}), 400

    # Ensure geo constraints are configured
    if att.geo_lat is None or att.geo_lng is None or not att.geo_radius:
        return jsonify({"success": False, "error": "Geo constraints not set"}), 400

    # Parse lat/lng from request
    data = request.get_json(silent=True) or {}
    try:
        user_lat = float(data.get("lat"))
        user_lng = float(data.get("lng"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "Invalid lat/lng"}), 400

    # Calculate distance
    dist = _haversine(user_lat, user_lng, att.geo_lat, att.geo_lng)

    if dist > float(att.geo_radius):
        return jsonify({
            "success": False,
            "error": "Outside allowed radius",
            "distance": dist
        }), 400

    # ✅ Update record
    record = _get_or_create_record(student_id, att.id)
    record.geo_valid = True
    required = _required_methods(att)
    record.update_status(required)
    db.session.commit()

    return jsonify({
        "success": True,
        "present": record.is_present,
        "distance": dist,
        "validated": {
            "qr": record.qr_valid,
            "geo": record.geo_valid,
            "face": record.face_valid,
        },
        "required_methods": required
    })



# ---------------- polling ----------------

@student_bp.route("/student/check_session/<int:attendance_id>")
def check_session(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"active": False, "error": "Not logged in"}), 401

    att = Attendance.query.get_or_404(attendance_id)
    record = _get_or_create_record(student_id, att.id)

    # Re-evaluate completion in case some validations already done
    record.update_status(_required_methods(att))
    db.session.commit()

    validations = {
        "qr": bool(record.qr_valid),
        "geo": bool(record.geo_valid),
        "face": bool(record.face_valid),
    }

    return jsonify({
        "active": att.is_active,
        "validation": validations,
        "required_methods": _required_methods(att),
        "present": record.is_present,
        # let UI hide QR block after success
        "hide_qr": bool(record.qr_valid),
    })






import numpy as np
import faiss
import cv2
from insightface.app import FaceAnalysis
import logging
log = logging.getLogger(__name__)
import os
FACE_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "face_db_faiss.npz")
FACE_DB_FILE = os.path.abspath(FACE_DB_FILE)
SIM_THRESHOLD = 0.5
_face_app = None
_face_names = []
_face_embs = None
_faiss_index = None


def init_face_app(det_size=(640, 640), provider="CPU"):
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0 if provider != "CPU" else -1, det_size=det_size)
        _face_names, _faiss_index = load_face_db()
        log.info("InsightFace app initialized.")
    return _face_app

# make sure compute_embedding_from_imagefile uses init_face_app()
def compute_embedding_from_imagefile(filename):
    """
    Read image from absolute filename and return 1D float32 normalized embedding, or None
    """
    try:
        app = init_face_app()
        img = cv2.imread(filename)
        if img is None:
            log.warning("Cannot read image file %s", filename)
            return None
        faces = app.get(img)
        if not faces:
            log.warning("No face found in image %s", filename)
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = f.normed_embedding
        if emb is None or emb.size == 0:
            log.warning("Empty embedding for file %s", filename)
            return None
        return emb.astype("float32")
    except Exception as e:
        log.exception("Error computing embedding for %s: %s", filename, e)
        return None

def get_embedding_from_image(img_np):
    """
    Given an OpenCV image (BGR), detect the largest face and return normalized embedding.
    Returns None if no face detected.
    """
    try:
        app = init_face_app()
        faces = app.get(img_np)
        if not faces:
            log.warning("No face detected in submitted image.")
            return None

        # pick the largest face
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        emb = f.normed_embedding
        if emb is None or emb.size == 0:
            log.warning("Empty embedding from submitted image.")
            return None

        return emb.astype("float32")
    except Exception as e:
        log.exception("Error computing embedding from image: %s", e)
        return None

def load_to_list():
    global _face_names, _face_embs, _faiss_index
    if not os.path.exists(FACE_DB_FILE):
        _face_names = []
        _face_embs = None
        _faiss_index = None
        return _face_names

    db = np.load(FACE_DB_FILE, allow_pickle=True)
    _face_names = db["names"].tolist()
    return _face_names

def load_face_db():
    """Load names and embeddings from FACE_DB_FILE. Rebuild FAISS index in memory."""
    global _face_names, _face_embs, _faiss_index
    if not os.path.exists(FACE_DB_FILE):
        _face_names = []
        _face_embs = None
        _faiss_index = None
        return _face_names, _faiss_index

    db = np.load(FACE_DB_FILE, allow_pickle=True)
    _face_names = db["names"].tolist()
    _face_embs = db["embs"].astype("float32")
    # build index
    dim = _face_embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (works with normalized embeddings)
    index.add(_face_embs)
    _faiss_index = index
    log.info("Loaded face DB with %d embeddings.", len(_face_names))
    return _face_names, _faiss_index

def save_face_db():
    """Persist current _face_names and _face_embs to disk."""
    global _face_names, _face_embs
    if _face_embs is None or len(_face_names) == 0:
        # save empty arrays to avoid crashes
        np.savez_compressed(FACE_DB_FILE, names=np.array([], dtype=object), embs=np.zeros((0, 512), dtype="float32"))
        return
    np.savez_compressed(FACE_DB_FILE, names=np.array(_face_names, dtype=object), embs=_face_embs)
    log.info("Saved face DB to %s", FACE_DB_FILE)

def rebuild_faiss_index():
    """Rebuild the in-memory FAISS index from _face_embs."""
    global _faiss_index, _face_embs
    if _face_embs is None or _face_embs.shape[0] == 0:
        _faiss_index = None
        return None
    dim = _face_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(_face_embs)
    _faiss_index = index
    return _faiss_index


_face_names, _faiss_index = load_face_db()
_face_app = init_face_app()

@student_bp.route("/student/face_status/<int:attendance_id>", methods=["GET"])
def face_status(attendance_id):
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"has_embedding": False}), 401

    student = Student.query.get(student_id)
    if not student:
        return jsonify({"has_embedding": False}), 404

    # Check if embedding exists in your FAISS DB (_face_names/_face_embs)
    global _face_names
    _face_names , _ = load_face_db()
    # if _face_names == []:
    #     try:
    #         _face_names , _ = load_face_db()
    #     except:
    #         pass
    print(_face_names)
    has_embedding = str(student.reg_no) in _face_names if _face_names else False

    return jsonify({"has_embedding": has_embedding})


import base64
import cv2
import numpy as np
from flask import request, jsonify
from deepface import DeepFace
import tempfile
from datasouls_antispoof.pre_trained_models import create_model
from datasouls_antispoof.class_mapping import class_mapping


# def is_real_face(img_np):
#     """
#     Checks if the given image (numpy array) is a real live face.
#     Returns True if real, False if spoofed or error.
#     """
#     try:
#         # Save the image temporarily because DeepFace expects a file path
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#             tmp_path = tmp.name
#             cv2.imwrite(tmp_path, img_np)
#         image_replay = load_rgb(tmp_path)
#         with torch.no_grad():
#             prediction = model(torch.unsqueeze(transform(image=image_replay)['image'], 0)).numpy()[0]
#         classes = list(class_mapping.keys())

#         # Get the index of the highest probability
#         predicted_index = prediction.argmax()  # or torch.argmax(torch.tensor(prediction))

#         # Get the class name
#         predicted_class = classes[predicted_index]

#         for cls, prob in zip(classes, prediction):
#             print(f"{cls}: {prob:.6f}")
#         # Run anti-spoofing check
#         # db_path can be any valid folder; we don't need database search here, just anti-spoofing
#         print("----------------------")
#         os.remove(tmp_path)

#         # If result is empty DataFrame, likely spoofed
#         return not result.empty

#     except Exception as e:
#         log.error(f"Error in spoof detection: {e}")
#         return False

@student_bp.route("/student/submit_face/<int:attendance_id>", methods=["POST"])
def submit_face(attendance_id):
    global _face_names, _face_embs, _faiss_index
    _face_names, _faiss_index = load_face_db()

    # Get logged-in student
    student_id = session.get("student_id")
    if not student_id:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    student = Student.query.get(student_id)
    if not student:
        return jsonify({"success": False, "error": "Student not found"}), 404

    # Get attendance session
    att = Attendance.query.get_or_404(attendance_id)
    if not att.is_active:
        return jsonify({"success": False, "error": "Session ended"}), 400

    # Decode image
    data = request.get_json()
    img_base64 = data.get("img_base64")
    if not img_base64:
        return jsonify({"success": False, "error": "No image received"}), 400

    try:
        img_str = img_base64.split(",")[-1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid image data: {e}"}), 400
    
    emb = get_embedding_from_image(img_np)
    if emb is None:
        return jsonify({"success": False, "error": "No face detected"}), 400

    # Validate against ONLY this student's embedding
    if not _face_names or _face_embs is None:
        return jsonify({"success": False, "error": "No embeddings in database"}), 500

    label, confidence = predict_image(img_np)
    print(label,confidence)
    if att.spoof_detection:
        label, confidence = predict_image(img_np)
        if label == "spoof" and confidence >= 0.9:
            return jsonify({"success": False, "error": "Spoofed face detected"}), 400

    try:
        idx = _face_names.index(str(student.reg_no))
    except ValueError:
        return jsonify({"success": False, "error": "No embedding found for this student"}), 404

    student_emb = _face_embs[idx]
    sim = float(np.dot(emb, student_emb))  # cosine similarity (embs are normalized)
    print(sim)
    if sim < SIM_THRESHOLD:
        return jsonify({"success": False, "error": "Face does not match registered student"}), 400

    # Success → mark attendance
    record = AttendanceRecord.query.filter_by(attendance_id=attendance_id, student_id=student_id).first()
    if not record:
        record = AttendanceRecord(attendance_id=attendance_id, student_id=student_id)
        db.session.add(record)

    record.face_valid = True
    record.update_status(att.methods.split(","))
    db.session.commit()

    return jsonify({"success": True, "present": record.is_present})



@student_bp.route("/profile")
@student_login_required
def profile():
    student = Student.query.get(session["student_id"])
    return render_template("student_profile.html", student=student)

# ---------- Password reset ----------
@student_bp.route("/profile/reset_password", methods=["GET", "POST"])
@student_login_required
def reset_password():
    student = Student.query.get(session["student_id"])
    if request.method == "POST":
        step = request.form.get("step")

        # Step 1: send OTP
        if step == "send_otp":
            otp = str(random.randint(100000, 999999))
            session["otp"] = otp
            session["otp_email"] = student.email

            # TODO: replace with real mail send function
            print(f"OTP for {student.email} is {otp}")

            flash("OTP has been sent to your registered email.", "info")
            return redirect(url_for("student.reset_password"))

        # Step 2: verify + reset
        if step == "verify":
            otp_entered = request.form.get("otp")
            new_password = request.form.get("new_password")

            if session.get("otp") == otp_entered and session.get("otp_email") == student.email:
                student.set_password(new_password)
                db.session.commit()

                session.pop("otp", None)
                session.pop("otp_email", None)

                flash("Password changed successfully.", "success")
                return redirect(url_for("student.profile"))
            else:
                flash("Invalid OTP.", "danger")

    return render_template("student_reset_password.html")

@student_bp.route('/student/quiz/list')
def student_quiz_list():
    student_id = session.get('student_id'); student = Student.query.get(student_id)
    my_classes = student.classes
    ongoing = Quiz.query.filter(Quiz.is_active==True, Quiz.class_id.in_([c.id for c in my_classes])).all()
    return render_template('student_quiz_list.html', ongoing=ongoing)

@student_bp.route('/student/quiz/take/<int:quiz_id>', methods=['GET'])
def take_quiz(quiz_id):
    student_id = session.get('student_id')
    q = Quiz.query.get_or_404(quiz_id)
    attempt = QuizAttempt.query.filter_by(quiz_id=q.id, student_id=student_id).first()

    if not q.is_active:
        flash("Quiz is closed", "warning")
        return redirect(url_for('student.student_quiz_list'))

    if attempt and attempt.submitted_at:
        flash("You have already submitted this quiz.", "info")
        return redirect(url_for('student.student_quiz_list'))
    # create or fetch attempt
    attempt = QuizAttempt.query.filter_by(quiz_id=q.id, student_id=student_id, submitted_at=None).first()
    if not attempt:
        attempt = QuizAttempt(quiz_id=q.id, student_id=student_id)
        db.session.add(attempt); db.session.commit()

    # show question by index param
    idx = int(request.args.get('q', 1))
    question = QuizQuestion.query.filter_by(quiz_id=q.id, index=idx).first()
    total = len(q.questions)
    return render_template('take_quiz.html', quiz=q, question=question, attempt=attempt, idx=idx, total=total)


@student_bp.route('/student/quiz/answer', methods=['POST'])
def submit_answer():
    student_id = session.get('student_id')
    attempt_id = int(request.form['attempt_id'])
    question_id = int(request.form['question_id'])
    selected_idx = request.form.get('selected_idx')

    attempt = QuizAttempt.query.get_or_404(attempt_id)
    question = QuizQuestion.query.get_or_404(question_id)

    selected_option = QuizOption.query.filter_by(question_id=question_id, idx_char=selected_idx).first()
    is_correct = bool(selected_option and selected_option.id == question.correct_option_id)

    existing = QuizAnswer.query.filter_by(attempt_id=attempt.id, question_id=question_id).first()
    if existing:
        existing.selected_option_id = selected_option.id if selected_option else None
        existing.is_correct = is_correct
    else:
        ans = QuizAnswer(
            attempt_id=attempt.id,
            question_id=question_id,
            selected_option_id=selected_option.id if selected_option else None,
            is_correct=is_correct
        )
        db.session.add(ans)
    db.session.commit()

    # figure out next question
    next_idx = question.index + 1
    if next_idx <= len(attempt.quiz.questions):
        return redirect(url_for('student.take_quiz', quiz_id=attempt.quiz.id, q=next_idx))
    else:
        return redirect(url_for('student.finish_attempt', attempt_id=attempt.id))


@student_bp.route('/student/quiz/submit/<int:attempt_id>', methods=['POST'])
def finish_attempt(attempt_id):
    attempt = QuizAttempt.query.get_or_404(attempt_id)
    if attempt.submitted_at:
        flash("Quiz already submitted.", "info")
        return redirect(url_for('student.student_quiz_list'))

    total_q = len(attempt.quiz.questions)
    correct = sum(1 for a in attempt.answers if a.is_correct)
    attempt.score = (correct / total_q * 100.0) if total_q > 0 else 0.0
    attempt.submitted_at = datetime.utcnow()
    db.session.commit()

    flash(f"Quiz submitted successfully! Score: {attempt.score:.2f}%", "success")
    return redirect(url_for('student.student_quiz_list'))
