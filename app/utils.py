from functools import wraps
from flask import session, redirect, url_for, flash

def student_login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("student_id"):
            flash("Please log in as a student first.", "warning")
            return redirect(url_for("student_auth.login"))
        return f(*args, **kwargs)
    return decorated_function