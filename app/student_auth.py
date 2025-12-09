from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from .models import Student
from flask_login import login_user
student_auth_bp = Blueprint("student_auth", __name__, url_prefix="/student")

# ----------------- Student Login -----------------
@student_auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        reg_no = request.form.get("reg_no", "").strip()
        password = request.form.get("password", "").strip()

        student = Student.query.filter_by(reg_no=reg_no).first()
        if student and student.check_password(password):
            session.clear()
            session["student_id"] = student.id
            session["role"] = "student"
            flash(f"Welcome {student.name}!", "success")
            return redirect(url_for("student.dashboard")) 

        flash("Invalid Register Number or Password", "danger")
        return redirect(url_for("student_auth.login"))

    return render_template("student_login.html")


# ----------------- Student Logout -----------------
@student_auth_bp.route("/logout")
def logout():
    session.pop("student_id", None)
    session.pop("role", None)
    flash("Logged out successfully", "info")
    return redirect(url_for("student_auth.login"))
