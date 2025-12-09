import os
from flask import Flask, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from config import Config, UPLOAD_DIR
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

# Globals
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = 'auth.login'


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure uploads dir exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)

    from .models import Teacher, Class, Attendance  # import models needed

    # Blueprints
    from .auth import auth_bp
    from .main import main_bp
    from .student import student_bp
    from .student_auth import student_auth_bp

    app.register_blueprint(student_auth_bp)
    app.register_blueprint(student_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    @login_manager.user_loader
    def load_user(user_id):
        from .models import Teacher, Student
        role, real_id = user_id.split(":")
        if role == "teacher":
            return Teacher.query.get(int(real_id))
        elif role == "student":
            return Student.query.get(int(real_id))
        return None

    @app.route('/uploads/students/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # -------------------------
    # AUTO ATTENDANCE SCHEDULER
    # -------------------------
    SLOT_TIMINGS = {
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
        "V3": ("20:02", "20:25"),
    }

    def start_auto_attendance():
        """Run every minute to check slots and start/close attendance sessions."""
        now = datetime.now()
        day = now.strftime("%a").upper()[:3]  # MON, TUE, SAT, SUN
        current_time = now.strftime("%H:%M")

        for slot_code, (start, end) in SLOT_TIMINGS.items():
            full_code = f"{day}_{slot_code}"

            with app.app_context():
                # 1) START attendance if inside slot time
                if start <= current_time <= end:
                    classes = Class.query.filter_by(auto_attendance=True).all()
                    for cls in classes:
                        if not cls.slots or full_code not in cls.slots.split(","):
                            continue

                        existing = Attendance.query.filter_by(
                            class_id=cls.id, date=now.date(), slot=slot_code
                        ).first()
                        if existing:
                            continue

                        # Build methods dynamically from class auto_* fields
                        methods = []
                        if cls.auto_qr_type:
                            methods.append("qr")
                        if cls.auto_geo_enabled:
                            methods.append("geo")
                        if cls.auto_face_enabled:
                            methods.append("face")
                        if not methods:
                            methods.append("manual")

                        att = Attendance(
                            class_id=cls.id,
                            date=now.date(),
                            slot=slot_code,
                            created_by_teacher=False,   # ✅ auto-created
                            methods=",".join(methods),
                            qr_type=cls.auto_qr_type,
                            qr_rotate_seconds=cls.auto_qr_rotate_seconds,
                            geo_lat=cls.auto_geo_lat,
                            geo_lng=cls.auto_geo_lng,
                            geo_radius=cls.auto_geo_radius,
                            spoof_detection=cls.auto_spoof_detection,
                            is_active=True,
                        )
                        db.session.add(att)
                        db.session.commit()
                        print(f"[AUTO] Started {cls.name} ({full_code}) using {att.methods}")

                # 2) CLOSE attendance if current time is past slot end
                if current_time > end:
                    active_attendances = Attendance.query.filter_by(
                        date=now.date(), slot=slot_code, is_active=True
                    ).all()
                    for att in active_attendances:
                        att.is_active = False
                        db.session.commit()
                        print(f"[AUTO] Closed attendance for class {att.class_id} ({full_code})")

    # Run scheduler only once (avoid multiple workers creating duplicates)
    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler = BackgroundScheduler()
        scheduler.add_job(start_auto_attendance, "interval", minutes=1)
        scheduler.start()

    return app
