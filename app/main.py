import os
from datetime import datetime, timedelta
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, send_file, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .models import Teacher, Student, Class, Attendance, AttendanceRecord , Mark , Quiz , QuizQuestion, QuizOption, QuizAttempt , QuizAnswer
from .models import AppConfig
from .auth import admin_required
import qrcode, io
import numpy as np
import faiss
import cv2
from sqlalchemy import or_
from insightface.app import FaceAnalysis
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time
from .services.quiz_generation import generate_quiz_from_file
log = logging.getLogger(__name__)


###################################
FACE_DB_FILE = os.path.join(os.path.dirname(__file__), "..", "face_db_faiss.npz")
FACE_DB_FILE = os.path.abspath(FACE_DB_FILE)
SIM_THRESHOLD = 0.32
_face_app = None
_face_names = []
_face_embs = None
_faiss_index = None

def superadmin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_superadmin:
            flash("Access denied: Superadmin only.", "danger")
            return redirect(url_for("main.dashboard"))
        return f(*args, **kwargs)
    return decorated_function


def init_face_app(det_size=(640, 640), provider="CPU"):
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name="buffalo_l")
        _face_app.prepare(ctx_id=0 if provider != "CPU" else -1, det_size=det_size)
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

# ---------- improved add_or_update_embedding ----------
def add_or_update_embedding(student, img_path=None):
    """
    Compute embedding for given student and add/update it in the face DB.
    - student: Student model instance (must have .reg_no)
    - img_path: optional absolute path to the image file. If not provided,
                this derives from current_app.config['UPLOAD_FOLDER'] + student.image_path
    Returns True if embedding added/updated, False otherwise.
    """
    global _face_names, _face_embs, _faiss_index

    # resolve absolute image path
    if img_path:
        abs_path = img_path
    else:
        if not getattr(student, "image_path", None):
            log.debug("Student %s has no image_path; skipping embedding", getattr(student, "reg_no", "<unknown>"))
            return False
        upload_folder = current_app.config.get("UPLOAD_FOLDER")
        if not upload_folder:
            log.error("UPLOAD_FOLDER not configured")
            return False
        abs_path = os.path.join(upload_folder, student.image_path)

    if not os.path.exists(abs_path):
        log.warning("Image file not found for student %s: %s", student.reg_no, abs_path)
        return False

    emb = compute_embedding_from_imagefile(abs_path)
    if emb is None:
        log.warning("No embedding extracted for %s from %s", student.reg_no, abs_path)
        return False

    # ensure face DB loaded
    if _face_embs is None and _faiss_index is None:
        load_face_db()

    label = str(student.reg_no)  # mapping uses reg_no (you can use id if you prefer)

    if _face_names and label in _face_names:
        idx = _face_names.index(label)
        # update existing embedding vector
        _face_embs[idx] = emb
        log.info("Updated embedding for %s at index %d", label, idx)
    else:
        # append new embedding
        if _face_embs is None:
            _face_embs = np.expand_dims(emb, axis=0)
        else:
            _face_embs = np.vstack([_face_embs, np.expand_dims(emb, axis=0)])
        if _face_names is None:
            _face_names = []
        _face_names.append(label)
        print("Embedding added")
        log.info("Added new embedding for %s", label)

    # rebuild index and persist
    rebuild_faiss_index()
    save_face_db()
    return True


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
load_face_db()
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



def remove_embedding_for_student(student):
    """
    Remove a student's embedding from DB (e.g., when deleting student).
    """
    global _face_names, _face_embs
    label = str(student.reg_no)

    if not _face_names or label not in _face_names:
        return False

    idx = _face_names.index(label)
    # Remove from list
    _face_names.pop(idx)

    # Remove from embeddings array safely
    if _face_embs is not None and _face_embs.shape[0] > 1:
        _face_embs = np.delete(_face_embs, idx, axis=0)
    else:
        _face_embs = None

    # Rebuild FAISS index
    rebuild_faiss_index()
    save_face_db()

    log.info("Removed embedding for %s", label)
    return True

######################################




main_bp = Blueprint('main', __name__)

# ---------------- Helper Functions -----------------

def get_current_qr(attendance: Attendance):
    if attendance.qr_type == 'rotate':
        now = datetime.utcnow()
        if not hasattr(attendance, '_last_qr_time'):
            attendance._last_qr_time = now
        if (now - attendance._last_qr_time).total_seconds() >= attendance.qr_rotate_seconds:
            attendance.generate_qr()
            attendance._last_qr_time = now
    elif not attendance.current_qr:
        attendance.generate_qr()
    return attendance.current_qr

def allowed_file(filename: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in current_app.config['ALLOWED_EXTENSIONS']

def allowed_file_quiz(filename: str) -> bool:
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in current_app.config['ALLOWED_EXTENSIONS_QUIZ']
# ---------------- Routes -----------------

@main_bp.route('/')
@login_required
def index():
    return redirect(url_for('main.dashboard'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Classes where the current user is the teacher
    my_classes = Class.query.filter_by(teacher_id=current_user.id).all()

    # For admins, get all classes
    all_classes = []
    if current_user.is_admin:
        all_classes = Class.query.all()

    # Get global automatic attendance flag (default false if not set yet)
    auto_att = AppConfig.get_value("automatic_attendance", "false")

    return render_template(
        'dashboard.html',
        my_classes=my_classes,
        all_classes=all_classes,
        auto_att=auto_att
    )



@main_bp.route("/toggle_automatic_attendance", methods=["POST"])
@login_required
def toggle_automatic_attendance():
    if not current_user.is_superadmin:
        flash("Unauthorized action.", "danger")
        return redirect(url_for("main.dashboard"))

    current_value = AppConfig.get_value("automatic_attendance", "false")
    new_value = "false" if current_value == "true" else "true"
    AppConfig.set_value("automatic_attendance", new_value)

    flash(f"Automatic Attendance set to {new_value.upper()}.", "success")
    return redirect(url_for("main.dashboard"))


import pytz
from datetime import datetime

@main_bp.app_template_filter("localtime")
def localtime_filter(dt, tzname="Asia/Kolkata", fmt="%Y-%m-%d %H:%M:%S"):
    if dt is None:
        return ""
    tz = pytz.timezone(tzname)
    return dt.replace(tzinfo=pytz.UTC).astimezone(tz).strftime(fmt)

# ---------------- Student Management -----------------
@main_bp.route('/students/add', methods=['GET', 'POST'])
@login_required
@admin_required
def add_student():
    if request.method == 'POST':
        reg_no = request.form.get('reg_no', '').strip()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        branch = request.form.get('branch', '').strip() or None
        specialization = request.form.get('specialization', '').strip() or None
        batch = request.form.get('batch', '').strip() or None
        image = request.files.get('image')

        if not reg_no or not name or not email or not password:
            flash('Reg No, Name, Email, and Password are required.', 'danger')
            return redirect(url_for('main.add_student'))

        if Student.query.filter_by(reg_no=reg_no).first():
            flash('A student with this Reg No already exists.', 'warning')
            return redirect(url_for('main.add_student'))

        if Student.query.filter_by(email=email).first():
            flash('A student with this email already exists.', 'warning')
            return redirect(url_for('main.add_student'))

        image_path = None
        save_path = None
        if image and image.filename:
            if not allowed_file(image.filename):
                flash('Only png/jpg/jpeg files are allowed.', 'danger')
                return redirect(url_for('main.add_student'))
            filename = secure_filename(f"{reg_no}_{image.filename}")
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            save_path = os.path.join(upload_folder, filename)
            image.save(save_path)
            image_path = filename

        # create student and commit (commit first so reg_no is guaranteed in DB)
        s = Student(
            reg_no=reg_no,
            name=name,
            email=email,
            branch=branch,
            specialization=specialization,
            batch=batch,
            image_path=image_path
        )
        s.set_password(password)
        db.session.add(s)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            log.exception("Failed to commit new student: %s", e)
            flash("Failed to add student.", "danger")
            return redirect(url_for('main.add_student'))

        # Now try to compute and add embedding (non-fatal)
        try:
            # pass absolute path if we have it (save_path), else add_or_update_embedding will derive from student.image_path
            added = add_or_update_embedding(s, img_path=save_path) if save_path else add_or_update_embedding(s)
            if not added:
                flash("Student added but face embedding was not created (no face detected or image issue).", "warning")
            else:
                flash("Student and face embedding added.", "success")
        except Exception as e:
            log.exception("Failed to generate embedding after student add: %s", e)
            flash("Student added but failed to generate face embedding.", "warning")

        return redirect(url_for('main.add_student'))

    return render_template('add_student.html')

@main_bp.route("/assign_admins")
@login_required
@superadmin_required
def assign_admins():
    teachers = Teacher.query.all()
    return render_template("assign_admins.html", teachers=teachers)


@main_bp.route("/toggle_admin/<int:teacher_id>")
@login_required
@superadmin_required
def toggle_admin(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.is_superadmin:
        flash("Cannot revoke superadmin rights.", "warning")
        return redirect(url_for("main.assign_admins"))

    teacher.is_admin = not teacher.is_admin
    db.session.commit()
    flash(f"{teacher.name} admin status set to {teacher.is_admin}.", "success")
    return redirect(url_for("main.assign_admins"))


@main_bp.route("/delete_teacher/<int:teacher_id>")
@login_required
@superadmin_required
def delete_teacher(teacher_id):
    teacher = Teacher.query.get_or_404(teacher_id)
    if teacher.is_superadmin:
        flash("Cannot delete a superadmin!", "danger")
        return redirect(url_for("main.assign_admins"))

    db.session.delete(teacher)
    db.session.commit()
    flash(f"Teacher {teacher.name} deleted.", "success")
    return redirect(url_for("main.assign_admins"))

import csv
from flask import Response

@main_bp.route('/students/template')
@login_required
@admin_required
def download_student_template():
    headers = ['reg_no', 'name', 'email', 'password', 'branch', 'specialization', 'batch']
    si = []
    si.append(','.join(headers))
    csv_str = '\n'.join(si)

    return Response(
        csv_str,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=student_template.csv"})

import io
import csv
from flask import request, flash, redirect, url_for, render_template
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.utils import secure_filename
import os




@main_bp.route('/students/bulk_add', methods=['POST'])
@login_required
@admin_required
def bulk_add_students():
    file = request.files.get('csv_file')
    if not file or not file.filename.endswith('.csv'):
        flash("Please upload a valid CSV file.", "danger")
        return redirect(url_for('main.add_student'))

    stream = io.StringIO(file.stream.read().decode("utf-8"))
    reader = csv.DictReader(stream)

    required_fields = ['reg_no', 'name', 'email', 'password']
    for field in required_fields:
        if field not in reader.fieldnames:
            flash(f"CSV missing required field: {field}", "danger")
            return redirect(url_for('main.add_student'))

    added = 0
    try:
        for row in reader:
            reg_no = row.get('reg_no', '').strip()
            name = row.get('name', '').strip()
            email = row.get('email', '').strip()
            password = row.get('password', '').strip()
            branch = row.get('branch', '').strip() or None
            specialization = row.get('specialization', '').strip() or None
            batch = row.get('batch', '').strip() or None

            if not reg_no or not name or not email or not password:
                raise ValueError(f"Missing required fields for reg_no={reg_no}")

            if Student.query.filter(
                (Student.reg_no == reg_no) | (Student.email == email)
            ).first():
                raise ValueError(f"Duplicate reg_no={reg_no} or email={email}")

            s = Student(
                reg_no=reg_no,
                name=name,
                email=email,
                branch=branch,
                specialization=specialization,
                batch=batch,
                image_path=None
            )
            s.set_password(password)
            db.session.add(s)
            added += 1

        db.session.commit()
        flash(f"Successfully added {added} students.", "success")
    except (ValueError, SQLAlchemyError) as e:
        db.session.rollback()
        flash(f"Upload failed: {str(e)}. No records were added.", "danger")

    return redirect(url_for('main.add_student'))



@main_bp.route('/students', methods=['GET'])
@login_required
@admin_required
def list_students():
    search = request.args.get('search', '').strip()
    branch = request.args.get('branch', '').strip()
    specialization = request.args.get('specialization', '').strip()
    batch = request.args.get('batch', '').strip()

    query = Student.query

    if search:
        query = query.filter(
            db.or_(
                Student.name.ilike(f'%{search}%'),
                Student.email.ilike(f'%{search}%'),
                Student.reg_no.ilike(f'%{search}%')
            )
        )
    if branch:
        query = query.filter(Student.branch.ilike(f'%{branch}%'))
    if specialization:
        query = query.filter(Student.specialization.ilike(f'%{specialization}%'))
    if batch:
        query = query.filter(Student.batch.ilike(f'%{batch}%'))

    students = query.all()
    return render_template(
        'list_students.html',
        students=students,
        search=search,
        branch=branch,
        specialization=specialization,
        batch=batch
    )


# Student attendance analytics page
@main_bp.route('/students/<int:student_id>/records')
@login_required
def view_student_records(student_id):
    # Permission: admins/superadmins OR teacher who teaches any class the student is in
    student = Student.query.get_or_404(student_id)

    # classes student is enrolled in
    student_classes = student.classes  # list of Class objects

    # Permission check: allow if current_user is admin/superadmin or teacher of at least one shared class
    allowed = False
    if getattr(current_user, "is_superadmin", False) or getattr(current_user, "is_admin", False):
        allowed = True
    else:
        # check if teacher of any class that the student is in
        teacher_class_ids = {c.id for c in getattr(current_user, "classes", [])}
        student_class_ids = {c.id for c in student_classes}
        if teacher_class_ids & student_class_ids:
            allowed = True

    if not allowed:
        flash("You do not have permission to view this student's records.", "danger")
        return redirect(url_for('main.list_students'))

    # Build per-class stats
    class_stats = []
    overall_present = 0
    overall_total = 0
    for cls in student_classes:
        sessions = Attendance.query.filter_by(class_id=cls.id).order_by(Attendance.created_at.asc()).all()
        total_sessions = len(sessions)
        # For each session determine whether student has a record marked present
        present_count = 0
        for s in sessions:
            rec = AttendanceRecord.query.filter_by(attendance_id=s.id, student_id=student.id).first()
            if rec and rec.is_present:
                present_count += 1
        percent = (present_count / total_sessions * 100) if total_sessions > 0 else None
        class_stats.append({
            "class_id": cls.id,
            "class_name": cls.name,
            "teacher_name": cls.teacher.name if cls.teacher else None,
            "total_sessions": total_sessions,
            "attended": present_count,
            "missed": total_sessions - present_count,
            "percent": round(percent, 2) if percent is not None else None
        })
        overall_present += present_count
        overall_total += total_sessions

    overall_percent = (overall_present / overall_total * 100) if overall_total > 0 else None

    return render_template(
        "student_records.html",
        student=student,
        class_stats=class_stats,
        overall_present=overall_present,
        overall_total=overall_total,
        overall_percent=round(overall_percent, 2) if overall_percent is not None else None
    )


# JSON endpoint returning per-session details for a student in a class (used by frontend charts)
@main_bp.route('/students/<int:student_id>/records/class/<int:class_id>/details')
@login_required
def student_class_details(student_id, class_id):
    student = Student.query.get_or_404(student_id)
    cls = Class.query.get_or_404(class_id)

    # Permission check same as above
    allowed = False
    if getattr(current_user, "is_superadmin", False) or getattr(current_user, "is_admin", False):
        allowed = True
    if not allowed:
        return jsonify({"error": "forbidden"}), 403

    sessions = Attendance.query.filter_by(class_id=class_id).order_by(Attendance.created_at.asc()).all()
    rows = []
    for s in sessions:
        rec = AttendanceRecord.query.filter_by(attendance_id=s.id, student_id=student.id).first()
        rows.append({
            "attendance_id": s.id,
            "date": s.created_at.strftime("%Y-%m-%d"),
            "datetime": s.created_at.isoformat(),
            "hour": s.created_at.hour,
            "present": bool(rec.is_present) if rec else False
        })

    return jsonify({
        "class_id": class_id,
        "class_name": cls.name,
        "student_id": student.id,
        "rows": rows
    })



@main_bp.route('/students/<int:student_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_student(student_id):
    s = Student.query.get_or_404(student_id)
    if request.method == 'POST':
        s.name = request.form.get('name', '').strip()
        s.email = request.form.get('email', '').strip()
        s.branch = request.form.get('branch', '').strip() or None
        s.specialization = request.form.get('specialization', '').strip() or None
        s.batch = request.form.get('batch', '').strip() or None

        password = request.form.get('password', '').strip()
        if password:
            s.set_password(password)

        image = request.files.get('image')
        save_path = None
        if image and image.filename:
            if not allowed_file(image.filename):
                flash('Only png/jpg/jpeg files are allowed.', 'danger')
                return redirect(url_for('main.edit_student', student_id=student_id))
            
            filename = secure_filename(f"{s.reg_no}_{image.filename}")
            upload_folder = current_app.config['UPLOAD_FOLDER']
            os.makedirs(upload_folder, exist_ok=True)
            save_path = os.path.join(upload_folder, filename)
            image.save(save_path)
            s.image_path = filename  # store only filename

            # ✅ Remove previous embedding if exists
            try:
                removed = remove_embedding_for_student(s)
                if removed:
                    print("Removed embedding of student")
            except:
                pass
        
        s.academics = request.form.get('academics', '').strip() or None
        s.co_curricular = request.form.get('co_curricular', '').strip() or None
        s.internships = request.form.get('internships', '').strip() or None
        s.remarks = request.form.get('remarks', '').strip() or None
        db.session.commit()

        # try to refresh embedding
        try:
            updated = add_or_update_embedding(s, img_path=save_path) if save_path else add_or_update_embedding(s)
            if updated:
                flash("Student updated and face embedding refreshed.", "success")
            else:
                flash("Student updated but face embedding not found/created.", "warning")
        except Exception as e:
            log.exception("Failed to update embedding: %s", e)
            flash("Student updated but failed to update face embedding.", "warning")

        return redirect(url_for('main.list_students'))

    return render_template('edit_student.html', student=s)




@main_bp.route('/admin/rebuild_face_db')
@login_required
@admin_required
def rebuild_face_db_route():
    students = Student.query.filter(Student.image_path != None).all()
    added = 0
    for s in students:
        ok = add_or_update_embedding(s)
        if ok:
            added += 1
    flash(f"Processed {len(students)} students, embeddings updated for {added}.", "success")
    return redirect(url_for('main.list_students'))



# ---------------- Class Management -----------------
@main_bp.route('/classes/create', methods=['GET', 'POST'])
@login_required
def create_class():
    all_students = Student.query.all()
    # ✅ teacher's occupied slots
    my_classes = Class.query.filter_by(teacher_id=current_user.id).all()
    occupied_slots = set()
    for cls in my_classes:
        if cls.slots:
            occupied_slots.update(cls.slots.split(","))

    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        batch = request.form.get('batch', '').strip()
        description = request.form.get('description', '').strip()
        selected_students = request.form.getlist('students')
        selected_slots = request.form.get('slots', '')

        if not name:
            flash('Class name is required.', 'danger')
            return redirect(url_for('main.create_class'))

        c = Class(name=name, batch=batch, description=description, teacher_id=current_user.id, slots=selected_slots)
        for sid in selected_students:
            st = Student.query.get(int(sid))
            if st:
                c.students.append(st)

        db.session.add(c)
        db.session.commit()
        flash('Class created successfully.', 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('create_class.html', students=all_students, occupied_slots=occupied_slots)


@main_bp.route('/classes/<int:class_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_class(class_id):
    # If superadmin -> can edit any class
    if current_user.is_superadmin:
        c = Class.query.get(class_id)
    else:
        # Only class teacher can edit their class
        c = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()

    if not c:
        flash('Class not found or you do not have permission to edit it.', 'warning')
        return redirect(url_for('main.dashboard'))

    all_students = Student.query.all()

    if request.method == 'POST':
        c.name = request.form.get('name', '').strip()
        c.batch = request.form.get('batch', '').strip()
        c.description = request.form.get('description', '').strip()

        # ✅ Save slots from hidden input
        c.slots = request.form.get("slots", "")

        # Update students
        selected_students = request.form.getlist('students')
        c.students = []
        for sid in selected_students:
            st = Student.query.get(int(sid))
            if st:
                c.students.append(st)

        db.session.commit()
        flash('Class updated successfully.', 'success')
        return redirect(url_for('main.dashboard'))

    return render_template('edit_class.html', cls=c, students=all_students)


@main_bp.route('/classes/<int:class_id>/assign', methods=['GET', 'POST'])
@login_required
def assign_students(class_id):
    # --- Permission logic ---
    if current_user.is_superadmin or current_user.is_admin:
        c = Class.query.get(class_id)  # full access
    else:
        c = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()

    if not c:
        flash('Class not found or you do not have permission to modify it.', 'warning')
        return redirect(url_for('main.dashboard'))

    # --- Filtering students ---
    search = request.args.get('search', '').strip()
    branch = request.args.get('branch', '').strip()
    specialization = request.args.get('specialization', '').strip()
    batch = request.args.get('batch', '').strip()

    query = Student.query
    if search:
        query = query.filter(
            db.or_(
                Student.name.ilike(f'%{search}%'),
                Student.reg_no.ilike(f'%{search}%'),
                Student.email.ilike(f'%{search}%')
            )
        )
    if branch:
        query = query.filter_by(branch=branch)
    if specialization:
        query = query.filter_by(specialization=specialization)
    if batch:
        query = query.filter_by(batch=batch)

    all_students = query.all()

    # --- Clash detection ---
    current_slots = set(c.slots.split(",")) if c.slots else set()
    clashes = {}  # {student_id: [(other_class_name, slot), ...]}
    for s in all_students:
        for other_cls in s.classes:
            if other_cls.id == c.id or not other_cls.slots:
                continue
            overlap = current_slots & set(other_cls.slots.split(","))
            if overlap:
                clashes[s.id] = [(other_cls.name, slot) for slot in overlap]

    # --- Handle assignment ---
    if request.method == 'POST':
        selected_students = request.form.getlist('students')
        c.students = []
        for sid in selected_students:
            st = Student.query.get(int(sid))
            # only allow if no clash
            if st and sid not in clashes:
                c.students.append(st)

        db.session.commit()
        flash('Students assigned successfully.', 'success')
        return redirect(url_for('main.dashboard'))

    return render_template(
        'assign_students.html',
        cls=c,
        students=all_students,
        search=search,
        branch=branch,
        specialization=specialization,
        batch=batch,
        clashes=clashes
    )



# ---------------- Attendance -----------------
@main_bp.route('/classes/<int:class_id>/attendance/create', methods=['GET', 'POST'])
@login_required
def create_attendance(class_id):
    c = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()
    if not c:
        flash("Class not found.", "danger")
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        methods = request.form.getlist('methods')
        if not methods:
            flash("Select at least one attendance method.", "warning")
            return redirect(url_for('main.create_attendance', class_id=class_id))
        methods_str = ",".join(methods)
        spoof_detection = 'spoof_detection' in request.form
        qr_type = request.form.get('qr_type') if 'qr' in methods else None
        qr_rotate_seconds = int(request.form.get('qr_rotate_seconds', 10)) if qr_type == 'rotate' else None

        geo_lat_str = request.form.get('geo_lat')
        geo_lat = float(geo_lat_str) if geo_lat_str else None
        geo_lng_str = request.form.get('geo_lng')
        geo_lng = float(geo_lng_str) if geo_lng_str else None
        geo_radius = int(request.form.get('geo_radius', 50)) if 'geo' in methods else None

        att = Attendance(
            class_id=c.id,
            methods=methods_str,
            qr_type=qr_type,
            qr_rotate_seconds=qr_rotate_seconds,
            geo_lat=geo_lat,
            geo_lng=geo_lng,
            geo_radius=geo_radius,
            spoof_detection=spoof_detection, 
            is_active=True,
            created_by_teacher=True
        )
        db.session.add(att)
        db.session.commit()

        for student in c.students:
            record = AttendanceRecord(
                attendance_id=att.id,
                student_id=student.id,
                is_present=False
            )
            db.session.add(record)
        db.session.commit()

        flash("Attendance session created.", "success")
        return redirect(url_for('main.attendance_session', attendance_id=att.id))

    return render_template('create_attendance.html', cls=c)


@main_bp.route('/classes/<int:class_id>/attendance', methods=['GET'])
@login_required
def view_attendance(class_id):
    # If superadmin or admin -> can view any class
    if current_user.is_superadmin or current_user.is_admin:
        c = Class.query.get(class_id)
    else:
        # Normal teacher -> only their classes
        c = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()

    if not c:
        flash('Class not found or not accessible.', 'warning')
        return redirect(url_for('main.dashboard'))

    sessions = Attendance.query.filter_by(class_id=c.id).order_by(Attendance.created_at.desc()).all()
    
    session_stats = []
    for s in sessions:
        total = len(s.records)
        present = sum(1 for r in s.records if r.is_present)
        absent = total - present
        session_stats.append({
            'attendance': s,
            'present': present,
            'absent': absent
        })

    return render_template('view_attendance.html', cls=c, session_stats=session_stats)



@main_bp.route('/attendance/<int:attendance_id>/records', methods=['GET'])
@login_required
def get_attendance_records(attendance_id):
    att = Attendance.query.get_or_404(attendance_id)
    records = []
    for r in att.records:
        records.append({
            'student_id': r.student.id,
            'name': r.student.name,
            'reg_no': r.student.reg_no,
            'is_present': r.is_present
        })
    return jsonify(records)


@main_bp.route('/attendance/<int:attendance_id>/update', methods=['POST'])
@login_required
def update_attendance(attendance_id):
    from .models import AttendanceLog

    data = request.get_json()
    student_id = data.get('student_id')
    present = data.get('present', False)

    record = AttendanceRecord.query.filter_by(attendance_id=attendance_id, student_id=student_id).first()
    if not record:
        return jsonify({'error': 'Record not found'}), 404

    # Only log if there is a change
    if record.is_present != present:
        record.is_present = present
        log = AttendanceLog(
            attendance_id=attendance_id,
            teacher_id=current_user.id,
            student_id=student_id,   # <--- log which student
            action=f"Marked { 'Present' if present else 'Absent' }"
        )
        db.session.add(log)


    db.session.commit()
    return jsonify({'present': record.is_present})



@main_bp.route('/attendance/session/<int:attendance_id>')
@login_required
def attendance_session(attendance_id):
    attendance = Attendance.query.get_or_404(attendance_id)
    cls = attendance.class_rel
    records = attendance.records
    return render_template(
        'attendance_session.html',
        attendance=attendance,
        cls=cls,
        students=[r.student for r in records],
        attendance_status={r.student.id: r.is_present for r in records}
    )


@main_bp.route('/attendance/<int:attendance_id>/qr')
@login_required
def get_qr(attendance_id):
    att = Attendance.query.get_or_404(attendance_id)
    qr_code = att.generate_qr()
    qr_data = url_for('main.validate_qr', qr_code=qr_code, attendance_id=att.id, _external=True)
    qr_img = qrcode.make(qr_data)
    buf = io.BytesIO()
    qr_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@main_bp.route('/attendance/<int:attendance_id>/validate/<qr_code>', methods=['GET', 'POST'])
@login_required
def validate_qr(attendance_id, qr_code):
    attendance = Attendance.query.get_or_404(attendance_id)
    if qr_code == attendance.current_qr:
        return "QR is valid", 200
    return "Invalid QR", 400


@main_bp.route('/attendance/<int:attendance_id>/end', methods=['POST'])
@login_required
def end_attendance(attendance_id):
    att = Attendance.query.get_or_404(attendance_id)
    att.is_active = False
    db.session.commit()
    return jsonify({'success': True})

@main_bp.route('/attendance/<int:attendance_id>/edit', methods=["GET", "POST"])
@login_required
def edit_attendance(attendance_id):
    from .models import AttendanceLog  # import inside if not already global

    attendance = Attendance.query.get_or_404(attendance_id)
    students = Student.query.all()
    cls = attendance.class_rel
    records = {r.student_id: r for r in attendance.records}  # map student_id → record

    if request.method == "POST":
        for student in students:
            status = request.form.get(f"student_{student.id}")  # checkbox/radio name in your form
            new_status = True if status == "present" else False

            rec = records.get(student.id)
            if rec:
                if rec.is_present != new_status:
                    # Update record
                    rec.is_present = new_status

                    # Log change
                    log = AttendanceLog(
                        attendance_id=attendance.id,
                        teacher_id=current_user.id,
                        student_id=student.id,
                        action=f"Changed to {'Present' if new_status else 'Absent'}"
                    )
                    db.session.add(log)
            else:
                # Create new record
                rec = AttendanceRecord(
                    attendance_id=attendance.id,
                    student_id=student.id,
                    is_present=new_status
                )
                db.session.add(rec)

                # Log creation
                log = AttendanceLog(
                    attendance_id=attendance.id,
                    teacher_id=current_user.id,
                    student_id=student.id,
                    action=f"Marked {'Present' if new_status else 'Absent'}"
                )
                db.session.add(log)

        db.session.commit()
        flash("Attendance updated successfully with logs.", "success")
        return redirect(url_for("main.edit_attendance", attendance_id=attendance.id))

    return render_template(
        "attendance_session.html",
        attendance=attendance,
        cls=cls,
        students=students,
        attendance_status={sid: rec.is_present for sid, rec in records.items()}
    )


@main_bp.route("/attendance/location/<int:attendance_id>")
def attendance_location(attendance_id):
    attendance = Attendance.query.get_or_404(attendance_id)
    lat = attendance.geo_lat
    lng = attendance.geo_lng
    address = "Not found"
    if lat and lng:
        try:
            url = f"https://nominatim.openstreetmap.org/reverse"
            params = {
                "format": "jsonv2",
                "lat": lat,
                "lon": lng
            }
            headers = {"User-Agent": "HalcyonAttendanceApp/1.0"}
            r = requests.get(url, params=params, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                address = data.get("display_name", "Not found")
        except Exception as e:
            address = "Error fetching"
    return jsonify({"address": address})


@main_bp.route("/attendance/<int:attendance_id>/records_all", methods=["GET", "POST"])
@login_required
def get_attendance_records_all(attendance_id):
    attendance = Attendance.query.get_or_404(attendance_id)

    # POST: toggle present
    if request.method == "POST":
        data = request.get_json()
        student_id = int(data.get("student_id"))
        present = data.get("present")

        record = AttendanceRecord.query.filter_by(attendance_id=attendance.id, student_id=student_id).first()
        if record:
            if record.is_present != bool(present):
                record.is_present = bool(present)
                record.timestamp = datetime.utcnow() if present else None

                log = AttendanceLog(
                    attendance_id=attendance.id,
                    teacher_id=current_user.id,
                    student_id=student_id,   # <--- log which student
                    action=f"Marked { 'Present' if present else 'Absent' }"
                )
                db.session.add(log)

            db.session.commit()


    # GET: return all records
    records = AttendanceRecord.query.filter_by(attendance_id=attendance.id).all()
    output = {}
    for r in records:
        output[str(r.student_id)] = {
            "student_id": r.student_id,
            "qr_valid": r.qr_valid,
            "geo_valid": r.geo_valid,
            "face_valid": r.face_valid,
            "is_present": r.is_present
        }
    return jsonify(output)




@main_bp.route("/attendance/<int:attendance_id>/toggle_active", methods=["POST"])
@login_required
def toggle_attendance_active(attendance_id):
    attendance = Attendance.query.get_or_404(attendance_id)
    # Flip is_active
    attendance.is_active = not attendance.is_active
    db.session.commit()
    return jsonify({"is_active": attendance.is_active})



@main_bp.route("/attendance/<int:attendance_id>/methods", methods=["GET"])
@login_required
def view_attendance_methods(attendance_id):
    attendance = Attendance.query.get_or_404(attendance_id)
    cls = attendance.cls
    students = Student.query.filter(Student.classes.contains(cls)).all()

    # Get current attendance status
    attendance_status = {}
    records = AttendanceRecord.query.filter_by(attendance_id=attendance.id).all()
    for r in records:
        attendance_status[r.student_id] = r.is_present

    # Determine methods dynamically
    methods = []
    if attendance.qr_type:
        methods.append('qr')
    if attendance.geo_lat and attendance.geo_lng:
        methods.append('geo')
    if attendance.face_enabled:
        methods.append('face')

    return render_template(
        "attendance_methods.html",
        cls=cls,
        attendance=attendance,
        students=students,
        attendance_status=attendance_status,
        methods=methods
    )

@main_bp.route("/attendance/<int:attendance_id>/methods_status", methods=["GET"])
@login_required
def attendance_methods_status(attendance_id):
    """
    Returns the validation status of all methods (qr, geo, face) for each student.
    """
    attendance = Attendance.query.get_or_404(attendance_id)

    # Determine methods dynamically (exclude 'manual')
    methods = []
    if attendance.methods:
        methods = [m for m in attendance.methods.split(",") if m != "manual"]

    records = AttendanceRecord.query.filter_by(attendance_id=attendance.id).all()
    output = {}
    for r in records:
        # Ensure final is_present is updated based on current validations
        r.update_status(methods)

        output[str(r.student_id)] = {
            "student_id": r.student_id,
            "is_present": r.is_present,
            "qr_valid": r.qr_valid,
            "geo_valid": r.geo_valid,
            "face_valid": r.face_valid
        }

    return jsonify({
        "methods": methods,   # list of active methods
        "records": output     # dict of student_id -> status
    })
    
@main_bp.route('/classes/<int:class_id>/face_stats')
@login_required
def get_face_stats(class_id):
    c = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()
    if not c:
        return jsonify({"error": "Class not found"}), 404
    
    all_faces = load_to_list()
    student_names = [s.name for s in c.students]

    with_embeddings = sum(1 for s in student_names if s in all_faces)
    total = len(student_names)
    without_embeddings = total - with_embeddings

    return jsonify({
        "total": total,
        "with_embeddings": with_embeddings,
        "without_embeddings": without_embeddings
    })

@main_bp.route('/toggle_attendance/<int:record_id>', methods=['POST'])
@login_required
def toggle_attendance(record_id):
    from .models import AttendanceLog

    record = AttendanceRecord.query.get_or_404(record_id)
    record.is_present = not record.is_present

    log = AttendanceLog(
        attendance_id=record.attendance_id,
        teacher_id=current_user.id,
        student_id=record.student_id,  # <--- log which student
        action=f"Toggled to { 'Present' if record.is_present else 'Absent' }"
    )
    db.session.add(log)


    db.session.commit()
    return jsonify({"success": True, "is_present": record.is_present})


# --- Toggle session active/inactive ---
@main_bp.route('/toggle_session/<int:attendance_id>', methods=['POST'])
@login_required
def toggle_session(attendance_id):
    session = Attendance.query.get_or_404(attendance_id)
    session.is_active = not session.is_active
    db.session.commit()
    return jsonify({"success": True, "is_active": session.is_active})

# --- API for session status ---
@main_bp.route('/get_session_status/<int:attendance_id>')
@login_required
def get_session_status(attendance_id):
    session = Attendance.query.get_or_404(attendance_id)
    return jsonify({"is_active": session.is_active})


@main_bp.route("/attendance/<int:attendance_id>/status")
@login_required
def get_attendance_status(attendance_id):
    records = AttendanceRecord.query.filter_by(attendance_id=attendance_id).all()
    data = []
    for record in records:
        # student has embedding if their reg_no exists in _face_names
        fn = load_to_list()
        has_embedding = str(record.student.reg_no) in fn
        data.append({
            "id": record.id,
            "student_id": record.student_id,
            "is_present": record.is_present,
            "has_embedding": has_embedding
        })
    return jsonify(data)

@main_bp.route("/students/status")
@login_required
def get_students_status():
    # Load face embeddings
    fn = load_to_list()  # uses your function

    # Get all students
    students = Student.query.all()
    data = []

    for s in students:
        has_embedding = str(s.reg_no) in fn
        data.append({
            "id": s.id,
            "name": s.name,
            "email": s.email,
            "reg_no": s.reg_no,
            "branch": s.branch,
            "specialization": s.specialization,
            "batch": s.batch,
            "image_path": s.image_path,
            "has_embedding": has_embedding
        })

    return jsonify(data)


@main_bp.route('/students/delete', methods=['POST'])
@login_required
@admin_required
def delete_student():
    data = request.get_json()
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'success': False, 'error': 'student_id missing'}), 400
    s = Student.query.get(student_id)
    if not s:
        return jsonify({'success': False, 'error': 'student not found'}), 404

    try:
        # try removing embedding (calls your helper)
        try:
            remove_embedding_for_student(s)
        except Exception as e:
            # log and continue - embedding removal shouldn't block delete
            current_app.logger.exception("Failed to remove embedding for student %s: %s", s.id, e)

        # remove student object
        db.session.delete(s)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        current_app.logger.exception("Error deleting student %s: %s", student_id, e)
        db.session.rollback()
        return jsonify({'success': False, 'error': 'delete failed'}), 500

# Bulk delete endpoint
@main_bp.route('/students/bulk-delete', methods=['POST'])
@login_required
@admin_required
def bulk_delete_students():
    data = request.get_json()
    ids = data.get('student_ids') or []
    if not isinstance(ids, (list, tuple)):
        return jsonify({'success': False, 'error': 'student_ids must be list'}), 400

    deleted = []
    failed = []
    for sid in ids:
        s = Student.query.get(sid)
        if not s:
            failed.append(sid)
            continue
        try:
            try:
                remove_embedding_for_student(s)
            except Exception as e:
                current_app.logger.exception("Failed to remove embedding for student %s: %s", sid, e)
            db.session.delete(s)
            deleted.append(sid)
        except Exception as e:
            current_app.logger.exception("Error deleting student %s: %s", sid, e)
            failed.append(sid)
    try:
        db.session.commit()
    except Exception as e:
        current_app.logger.exception("Error committing bulk delete: %s", e)
        db.session.rollback()
        return jsonify({'success': False, 'error': 'commit failed'}), 500

    return jsonify({'success': True, 'deleted': deleted, 'failed': failed})


@main_bp.route('/students/paginated', methods=['GET'])
@login_required
@admin_required
def list_students_paginated():
    search = request.args.get('search', '').strip()
    branch = request.args.get('branch', '').strip()
    specialization = request.args.get('specialization', '').strip()
    batch = request.args.get('batch', '').strip()
    per_page = int(request.args.get('per_page', 10))  # default 10
    page = int(request.args.get('page', 1))

    query = Student.query

    if search:
        query = query.filter(
            db.or_(
                Student.name.ilike(f'%{search}%'),
                Student.email.ilike(f'%{search}%'),
                Student.reg_no.ilike(f'%{search}%')
            )
        )
    if branch:
        query = query.filter(Student.branch.ilike(f'%{branch}%'))
    if specialization:
        query = query.filter(Student.specialization.ilike(f'%{specialization}%'))
    if batch:
        query = query.filter(Student.batch.ilike(f'%{batch}%'))

    students_pag = query.paginate(page=page, per_page=per_page, error_out=False)

    return render_template(
        'students_paginated.html',
        students=students_pag.items,
        students_pag=students_pag,
        search=search,
        branch=branch,
        specialization=specialization,
        batch=batch,
        per_page=per_page
    )


@main_bp.route('/classes/<int:class_id>/auto_attendance', methods=['GET', 'POST'])
@login_required
def auto_attendance_setup(class_id):
    cls = Class.query.get_or_404(class_id)

    # only allow superadmin/admin/teacher of class
    if not (current_user.is_superadmin or current_user.is_admin or cls.teacher_id == current_user.id):
        flash("You don’t have permission to edit this class.", "danger")
        return redirect(url_for('main.dashboard'))

    if request.method == 'POST':
        # master toggle
        cls.auto_attendance = 'auto_attendance' in request.form

        # QR config
        cls.auto_qr_type = request.form.get('auto_qr_type') or None
        cls.auto_qr_rotate_seconds = int(request.form.get('auto_qr_rotate_seconds') or 10)

        # Geo config
        cls.auto_geo_enabled = 'auto_geo_enabled' in request.form
        cls.auto_geo_lat = float(request.form.get('auto_geo_lat') or 0)
        cls.auto_geo_lng = float(request.form.get('auto_geo_lng') or 0)
        cls.auto_geo_radius = int(request.form.get('auto_geo_radius') or 50)

        # Face recognition
        cls.auto_face_enabled = 'auto_face_enabled' in request.form

        # ✅ Spoof detection (fix)
        cls.auto_spoof_detection = 'auto_spoof_detection' in request.form

        db.session.commit()
        flash("Auto attendance settings updated successfully.", "success")
        return redirect(url_for('main.auto_attendance_setup', class_id=cls.id))

    return render_template("auto_attendance_setup.html", cls=cls)



def start_auto_attendance():
    now = datetime.now()
    day = now.strftime("%a").upper()[:3]  # MON, TUE, etc
    current_time = now.strftime("%H:%M")  # 24hr string

    # Your slot definitions
    slots = {
        "A1": ("08:00", "08:50"),
        "F1": ("08:55", "09:45"),
        "D1": ("09:50", "10:40"),
        "TB1": ("11:45", "12:30"),
        "TG1": ("12:35", "13:25"),
        "A2": ("14:00", "14:50"),
        "F2": ("14:55", "15:45"),
        "D2": ("15:50", "16:40"),
        "TB2": ("16:45", "17:35"),
        "TG2": ("17:40", "18:30"),
        "V3": ("18:35", "19:25"),
    }

    for slot_code, (start, end) in slots.items():
        if start <= current_time <= end:
            full_code = f"{day}_{slot_code}"

            classes = Class.query.filter_by(auto_attendance=True).all()
            for cls in classes:
                if cls.slots and full_code in cls.slots.split(","):
                    # Check if already active attendance exists
                    existing = Attendance.query.filter_by(class_id=cls.id, is_active=True).first()
                    if not existing:
                        att = Attendance(
                            class_id=cls.id,
                            methods=",".join([
                                "manual",
                                "qr" if cls.auto_qr_type else "",
                                "geo" if cls.auto_geo_enabled else "",
                                "face" if cls.auto_face_enabled else ""
                            ]).strip(","),
                            qr_type=cls.auto_qr_type,
                            qr_rotate_seconds=cls.auto_qr_rotate_seconds,
                            geo_lat=cls.auto_geo_lat,
                            geo_lng=cls.auto_geo_lng,
                            geo_radius=cls.auto_geo_radius,
                            spoof_detection=cls.auto_spoof_detection,
                            is_active=True,
                            created_by_teacher=False
                        )
                        db.session.add(att)
                        db.session.commit()
                        print(f"[AUTO] Started attendance for {cls.name} ({full_code})")

@main_bp.route('/attendance/<int:attendance_id>/logs')
@login_required
def attendance_logs(attendance_id):
    from .models import AttendanceLog
    logs = AttendanceLog.query.filter_by(attendance_id=attendance_id).order_by(AttendanceLog.timestamp.desc()).all()
    att = Attendance.query.get_or_404(attendance_id)
    return render_template('attendance_logs.html', logs=logs, attendance=att)


@main_bp.route("/classes/<int:class_id>/marks", methods=["GET"])
@login_required
def class_marks(class_id):
    cls = Class.query.get_or_404(class_id)
    students = cls.students

    # find all unique exam names in this class
    exams = sorted({m.exam_name for m in cls.marks})

    # build a dict: {student_id: {exam_name: mark}}
    marks_data = {}
    for m in cls.marks:
        marks_data.setdefault(m.student_id, {})[m.exam_name] = m.mark

    return render_template("class_marks.html", cls=cls, students=students, exams=exams, marks_data=marks_data)


@main_bp.route("/classes/<int:class_id>/marks/update", methods=["GET", "POST"])
@login_required
def update_marks(class_id):
    cls = Class.query.get_or_404(class_id)
    students = cls.students

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            import csv, io
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            reader = csv.DictReader(stream)

            # exam names from CSV headers (excluding reg_no)
            exams = [h for h in reader.fieldnames if h.lower() != "reg_no"]

            valid_regnos = {s.reg_no: s.id for s in students}

            for row in reader:
                reg_no = row.get("reg_no")
                if reg_no not in valid_regnos:
                    continue  # skip invalid

                student_id = valid_regnos[reg_no]
                for exam in exams:
                    raw_mark = row.get(exam, "").strip()
                    mark_val = raw_mark if raw_mark else "Not Available"

                    # update or insert
                    m = Mark.query.filter_by(class_id=cls.id, student_id=student_id, exam_name=exam).first()
                    if not m:
                        m = Mark(class_id=cls.id, student_id=student_id, exam_name=exam)
                        db.session.add(m)
                    m.mark = mark_val
            db.session.commit()
            flash("Marks updated successfully!", "success")
            return redirect(url_for("main.class_marks", class_id=cls.id))

    return render_template("update_marks.html", cls=cls, students=students)


@main_bp.route("/classes/<int:class_id>/marks/download")
@login_required
def download_marks_csv(class_id):
    import csv, io
    cls = Class.query.get_or_404(class_id)
    students = cls.students

    exams = sorted({m.exam_name for m in cls.marks}) or ["exam_1"]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["reg_no"] + exams)

    for s in students:
        row = [s.reg_no]
        for exam in exams:
            m = Mark.query.filter_by(class_id=cls.id, student_id=s.id, exam_name=exam).first()
            row.append(m.mark if m else "")
        writer.writerow(row)

    output.seek(0)
    from flask import Response
    return Response(output, mimetype="text/csv", headers={"Content-Disposition":"attachment;filename=marks.csv"})

# inside main.py or a new blueprint
from flask import request, flash, redirect, url_for, render_template, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import os, uuid

@main_bp.route('/classes/<int:class_id>/quiz/create', methods=['GET','POST'])
@login_required
def create_quiz(class_id):
    cls = Class.query.filter_by(id=class_id, teacher_id=current_user.id).first()
    if not cls:
        flash("Class not found.", "danger")
        return redirect(url_for('main.dashboard'))

    if request.method == "POST":
        # --- Form Data ---
        prompt = request.form.get('prompt','Generate Quiz').strip()
        num_questions = int(request.form.get('num_questions', 10))
        difficulty = request.form.get('difficulty','Medium')
        time_limit_min = int(request.form.get('time_limit_min', 0))
        
        # --- Datetime values ---
        open_at_str = request.form.get("open_at")
        close_at_str = request.form.get("close_at")
        try:
            open_at = datetime.strptime(open_at_str, "%Y-%m-%dT%H:%M")
            close_at = datetime.strptime(close_at_str, "%Y-%m-%dT%H:%M")
        except Exception:
            flash("Invalid open/close date format", "danger")
            return redirect(url_for('main.create_quiz', class_id=class_id))

        # --- File upload ---
        f = request.files.get('source_file')
        filename = None
        file_path = None
        if f and f.filename:
            if not allowed_file_quiz(f.filename):
                flash("Invalid file type", "danger")
                return redirect(url_for('main.create_quiz', class_id=class_id))
            filename = secure_filename(f"{uuid.uuid4().hex}_{f.filename}")
            upload_folder = current_app.config['UPLOAD_FOLDER_QUIZ']
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, filename)
            f.save(file_path)

        # --- Create quiz DB record ---
        q = Quiz(
            class_id=cls.id,
            title=f"{cls.name} - quiz",
            prompt=prompt,
            source_file=filename,
            num_questions=num_questions,
            difficulty=difficulty,
            time_limit_seconds=time_limit_min*60,
            is_active=False,
            open_at=open_at,
            close_at=close_at
        )
        db.session.add(q)
        db.session.commit()

        # --- Generate questions ---
        parsed = generate_quiz_from_file(file_path or "", teacher_prompt=prompt, num_questions=num_questions, difficulty=difficulty)

        # --- Save questions & options ---
        created_qs = []
        for idx, item in enumerate(parsed.get('questions', [])):
            qst = QuizQuestion(
                quiz_id=q.id,
                index=idx+1,
                stem=item['stem'],
                difficulty=item.get('difficulty', difficulty),
                explanation=item.get('explanation')
            )
            db.session.add(qst)
            db.session.flush()  # get qst.id

            # Create options A..D
            for opt_i, opt_text in enumerate(item['options']):
                opt = QuizOption(
                    question_id=qst.id,
                    idx_char=chr(ord('A')+opt_i),
                    text=opt_text
                )
                db.session.add(opt)
                db.session.flush()
                if opt_i == int(item['answer_index']):
                    qst.correct_option_id = opt.id

            created_qs.append(qst)

        db.session.commit()

        flash("Quiz generated and saved. You can open it when ready.", "success")
        return redirect(url_for('main.quiz_session', quiz_id=q.id))

    return render_template('create_quiz.html', cls=cls)


@main_bp.route('/quiz/<int:quiz_id>/start', methods=['POST'])
@login_required
def start_quiz(quiz_id):
    q = Quiz.query.get_or_404(quiz_id)
    if q.class_rel.teacher_id != current_user.id:
        flash("❌ You are not authorized to start this quiz.", "danger")
        return redirect(url_for('main.dashboard'))
    q.is_active = True
    db.session.commit()
    flash(f"✅ Quiz '{q.title}' has been started successfully.", "success")
    return redirect(url_for('main.dashboard'))

@main_bp.route('/quiz/<int:quiz_id>/end', methods=['POST'])
@login_required
def end_quiz(quiz_id):
    q = Quiz.query.get_or_404(quiz_id)
    if q.class_rel.teacher_id != current_user.id:
        flash("❌ You are not authorized to end this quiz.", "danger")
        return redirect(url_for('main.dashboard'))
    q.is_active = False
    db.session.commit()
    flash(f"⚠️ Quiz '{q.title}' has been ended.", "warning")
    return redirect(url_for('main.dashboard'))

@main_bp.route('/quiz/session/<int:quiz_id>')
@login_required
def quiz_session(quiz_id):
    q = Quiz.query.get_or_404(quiz_id)
    # Render teacher preview page (list of questions, edit if needed)
    return render_template('quiz_session.html', quiz=q)
