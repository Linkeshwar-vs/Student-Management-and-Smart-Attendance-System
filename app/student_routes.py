from flask import Blueprint, render_template, session, redirect, url_for, flash, request
from .models import Student, Attendance, AttendanceRecord, Class
from . import db

student_bp = Blueprint("student", __name__, url_prefix="/student")


# --- Helpers ---
def get_current_student():
    sid = session.get("student_id")
    if not sid:
        return None
    return Student.query.get(sid)


# --- Middleware (protect all student routes) ---
@student_bp.before_request
def require_login():
    if not session.get("student_id") and request.endpoint != "student_auth.login":
        return redirect(url_for("student_auth.login"))


# --- Dashboard ---
@student_bp.route("/dashboard")
def dashboard():
    student = get_current_student()
    if not student:
        return redirect(url_for("student_auth.login"))

    # Ongoing attendance sessions for student’s classes
    ongoing_attendance = Attendance.query.filter(
        Attendance.class_id.in_([c.id for c in student.classes]),
        Attendance.is_active.is_(True)
    ).all()

    return render_template(
        "student_dashboard.html",
        student=student,
        ongoing_attendance=ongoing_attendance,
        my_classes=student.classes
    )


# --- Join class ---
@student_bp.route("/join_class", methods=["POST"])
def join_class():
    student = get_current_student()
    if not student:
        return redirect(url_for("student_auth.login"))

    class_code = request.form.get("class_code")
    classroom = Class.query.filter_by(code=class_code).first()

    if classroom:
        if classroom not in student.classes:
            student.classes.append(classroom)
            db.session.commit()
            flash("Successfully joined the class!", "success")
        else:
            flash("You are already in this class.", "info")
    else:
        flash("Invalid class code.", "danger")

    return redirect(url_for("student.dashboard"))


# --- Mark attendance ---
@student_bp.route("/attendance/<int:attendance_id>/mark", methods=["POST"])
def mark_attendance(attendance_id):
    student = get_current_student()
    if not student:
        return redirect(url_for("student_auth.login"))

    attendance = Attendance.query.get_or_404(attendance_id)

    # Check if already marked
    existing = AttendanceRecord.query.filter_by(
        attendance_id=attendance.id,
        student_id=student.id
    ).first()

    if existing:
        flash("You have already marked attendance.", "info")
    else:
        from datetime import datetime
        record = AttendanceRecord(
            attendance_id=attendance.id,
            student_id=student.id,
            is_present=True,
            timestamp=datetime.utcnow()
        )
        db.session.add(record)
        db.session.commit()
        flash("Attendance marked successfully!", "success")

    return redirect(url_for("student.dashboard"))


# --- View attendance records ---
@student_bp.route("/attendance/<int:class_id>")
def view_attendance(class_id):
    student = get_current_student()
    if not student:
        return redirect(url_for("student_auth.login"))

    records = AttendanceRecord.query.filter_by(student_id=student.id).join(Attendance).filter(
        Attendance.class_id == class_id
    ).all()

    return render_template(
        "student_attendance.html",
        student=student,
        records=records,
        class_id=class_id
    )
