from flask import Blueprint, render_template, request, redirect, url_for, flash,session
from flask_login import login_user, logout_user, login_required, current_user
from . import db
from .models import Teacher
from werkzeug.security import generate_password_hash

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        is_admin = request.form.get('is_admin') == 'true'

        # Check if email already exists
        if Teacher.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('auth.register'))

        # Create new teacher with name
        new_user = Teacher(
            name=name,
            email=email,
            is_admin=is_admin
        )
        new_user.set_password(password)  # your Teacher model must have set_password()

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = Teacher.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            session["role"] = "teacher"   # 👈 Store role in session
            flash('Logged in successfully.', 'success')
            next_url = request.args.get('next') or url_for('main.index')
            return redirect(next_url)

        flash('Invalid credentials', 'danger')
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out.', 'info')
    return redirect(url_for('auth.login'))


# Simple decorator to restrict to admins
from functools import wraps

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            flash('You are not authorized to access this page.', 'warning')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    return wrapper