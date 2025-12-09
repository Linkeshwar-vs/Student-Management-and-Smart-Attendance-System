from flask import Blueprint, render_template, redirect, url_for, request, flash, session
from werkzeug.security import check_password_hash
from .models import Student
from . import db

student_auth_bp = Blueprint('student_auth', __name__)

@student_auth_bp.route('/student/login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        reg_no = request.form['reg_no']
        password = request.form['password']

        student = Student.query.filter_by(reg_no=reg_no).first()
        if student and student.check_password(password):
            # store student id in session
            session['student_id'] = student.id
            return redirect(url_for('student.dashboard'))
        else:
            flash("Invalid Register Number or Password", "danger")

    return render_template('student_login.html')


@student_auth_bp.route('/student/logout')
def student_logout():
    session.pop('student_id', None)
    return redirect(url_for('student_auth.student_login'))
