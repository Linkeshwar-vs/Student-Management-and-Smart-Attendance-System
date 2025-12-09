from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from . import db

# models.py
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
    "V3": ("18:35", "19:25"),
}


# Association table: many-to-many Class <-> Student
class_student = db.Table(
    'class_student',
    db.Column('class_id', db.Integer, db.ForeignKey('class.id'), primary_key=True),
    db.Column('student_id', db.Integer, db.ForeignKey('student.id'), primary_key=True),
)

# ---------------- Teacher -----------------
class Teacher(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    is_admin = db.Column(db.Boolean, default=False)
    is_superadmin = db.Column(db.Boolean, default=False)  # new field

    classes = db.relationship('Class', backref='teacher', lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)
    def get_id(self):
        return f"teacher:{self.id}"
    

# ---------------- Student -----------------

class Student(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reg_no = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    image_path = db.Column(db.String(255))  # path under uploads/students

    # Optional fields
    branch = db.Column(db.String(100), nullable=True)
    specialization = db.Column(db.String(100), nullable=True)
    batch = db.Column(db.String(50), nullable=True)
    academics = db.Column(db.Text, nullable=True)        # academic info
    co_curricular = db.Column(db.Text, nullable=True)    # extra activities
    internships = db.Column(db.Text, nullable=True)      # internships
    remarks = db.Column(db.Text, nullable=True)          # remarks/notes
    
    classes = db.relationship('Class', secondary=class_student, backref='students')

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return f"student:{self.id}"

# ---------------- Class -----------------
class Class(db.Model):  # "Course"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    batch = db.Column(db.String(50))
    description = db.Column(db.String(255))

    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)
    slots = db.Column(db.String)
    # Relationships
    auto_attendance = db.Column(db.Boolean, default=False)

    auto_qr_type = db.Column(db.String(20))      # "static" or "rotate"
    auto_qr_rotate_seconds = db.Column(db.Integer, default=10)

    auto_geo_enabled = db.Column(db.Boolean, default=False)
    auto_geo_lat = db.Column(db.Float)
    auto_geo_lng = db.Column(db.Float)
    auto_geo_radius = db.Column(db.Integer, default=50)

    auto_face_enabled = db.Column(db.Boolean, default=False)
    auto_spoof_detection = db.Column(db.Boolean, default=False)

    attendance_sessions = db.relationship(
        'Attendance',
        backref=db.backref('class_rel', lazy=True),
        lazy=True,
        overlaps="attendance_sessions,class_rel"
    )

# ---------------- Attendance -----------------
class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    date = db.Column(db.Date, default=datetime.utcnow().date)   # ✅ session date
    slot = db.Column(db.String(10), nullable=True)              # ✅ slot code (A1, F1, etc.)
    created_by_teacher = db.Column(db.Boolean, default=True)
    methods = db.Column(db.String)  # e.g., "manual,qr,geo,face"
    qr_type = db.Column(db.String)  # "static" or "rotate"
    qr_rotate_seconds = db.Column(db.Integer, default=10)  # interval in seconds
    current_qr = db.Column(db.String(8), nullable=True)  # 8-digit QR code
    qr_generated_at = db.Column(db.DateTime, nullable=True)

    geo_lat = db.Column(db.Float)
    geo_lng = db.Column(db.Float)
    geo_radius = db.Column(db.Integer)  # meters
    spoof_detection = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)

    # Relationships
    records = db.relationship('AttendanceRecord', backref='attendance', lazy=True)

    def generate_qr(self):
        """Generate a new 8-digit QR code if attendance is active."""
        import random
        import string
        if self.is_active and self.qr_type == 'rotate':
            self.current_qr = ''.join(random.choices(string.digits, k=8))
            db.session.commit()
        elif not self.current_qr:
            # fallback for static QR
            self.current_qr = ''.join(random.choices(string.digits, k=8))
            db.session.commit()
        return self.current_qr
    def check_auto_close(self, max_duration_minutes=60):
        """Automatically close if session is older than max_duration_minutes."""
        if self.is_active and datetime.utcnow() - self.created_at > timedelta(minutes=max_duration_minutes):
            self.is_active = False
            db.session.commit()

from typing import List
# ---------------- AttendanceRecord -----------------
class AttendanceRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    
    # validation flags
    qr_valid = db.Column(db.Boolean, default=False)
    geo_valid = db.Column(db.Boolean, default=False)
    face_valid = db.Column(db.Boolean, default=False)

    # final status
    is_present = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime)

    student = db.relationship('Student', lazy=True)

    def update_status(self, required_methods: List[str]):
        """
        Check if all required methods are validated.
        If so, mark student present and set timestamp.
        """
        if all([
            ("qr" not in required_methods or self.qr_valid),
            ("geo" not in required_methods or self.geo_valid),
            ("face" not in required_methods or self.face_valid),
        ]):
            self.is_present = True
            from datetime import datetime
            self.timestamp = datetime.utcnow()

class AppConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)
    value = db.Column(db.String(50), nullable=False)

    @staticmethod
    def get_value(key, default=None):
        cfg = AppConfig.query.filter_by(key=key).first()
        return cfg.value if cfg else default

    @staticmethod
    def set_value(key, value):
        cfg = AppConfig.query.filter_by(key=key).first()
        if not cfg:
            cfg = AppConfig(key=key, value=value)
            db.session.add(cfg)
        else:
            cfg.value = value
        db.session.commit()
class AttendanceLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teacher.id'), nullable=False)
    action = db.Column(db.String(255))
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    teacher = db.relationship("Teacher", backref="attendance_logs", lazy=True)
    student = db.relationship("Student", backref="attendance_logs", lazy=True)
    attendance = db.relationship("Attendance", backref="logs", lazy=True)

class Mark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    exam_name = db.Column(db.String(100), nullable=False)   # e.g. "exam_1", "midterm", "final"
    mark = db.Column(db.String(20))  # can store "Abs", "Not Available", or numeric marks
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    student = db.relationship("Student", backref="marks", lazy=True)
    class_rel = db.relationship("Class", backref="marks", lazy=True)

class Quiz(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    prompt = db.Column(db.Text)                # teacher prompt (default "Generate Quiz")
    source_file = db.Column(db.String(255))    # saved filename (optional)
    num_questions = db.Column(db.Integer, default=10)
    difficulty = db.Column(db.String(32))      # Easy/Medium/Intermediate/Hard/Very Hard
    time_limit_seconds = db.Column(db.Integer, default=0)  # 0 = no limit
    created_by_teacher = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)  # teacher opens/closes quiz
    open_at = db.Column(db.DateTime, nullable=True)
    close_at = db.Column(db.DateTime, nullable=True)
    @property
    def start_time(self):
        return self.open_at

    @property
    def end_time(self):
        return self.close_at

    # relationships
    questions = db.relationship('QuizQuestion', backref='quiz', lazy=False)
    attempts = db.relationship('QuizAttempt', backref='quiz', lazy=False)

    class_rel = db.relationship("Class", backref="quizzes", lazy=True)

class QuizQuestion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    index = db.Column(db.Integer, nullable=False)   # question order
    stem = db.Column(db.Text, nullable=False)
    difficulty = db.Column(db.String(32))
    explanation = db.Column(db.Text, nullable=True)

    options = db.relationship('QuizOption', backref='question', lazy=True, cascade="all, delete-orphan")
    correct_option_id = db.Column(db.Integer)  # store id of correct QuizOption

class QuizOption(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('quiz_question.id'), nullable=False)
    idx_char = db.Column(db.String(1))  # 'A','B','C','D'
    text = db.Column(db.Text, nullable=False)

class QuizAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    submitted_at = db.Column(db.DateTime, nullable=True)
    score = db.Column(db.Float, nullable=True)  # percentage or points

    answers = db.relationship('QuizAnswer', backref='attempt', lazy=True, cascade="all, delete-orphan")

    student = db.relationship("Student", lazy=True)

class QuizAnswer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attempt_id = db.Column(db.Integer, db.ForeignKey('quiz_attempt.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('quiz_question.id'), nullable=False)
    selected_option_id = db.Column(db.Integer, nullable=True)
    is_correct = db.Column(db.Boolean, default=False)