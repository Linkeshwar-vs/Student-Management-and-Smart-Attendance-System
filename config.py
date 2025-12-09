import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "students")
# config / current_app.config


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or "supersecretkey"

    # PostgreSQL connection (update user/pass/db if needed)
    SQLALCHEMY_DATABASE_URI = (
        "postgresql://attendance_user:123@localhost:5432/attendance_db"
    )

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File upload settings
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB
    UPLOAD_FOLDER = UPLOAD_DIR
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    ALLOWED_EXTENSIONS_QUIZ = {'png','jpg','jpeg','pdf','docx','pptx','txt','md'}
    MAX_CONTENT_LENGTH_QUIZ = 10 * 1024 * 1024  # 10 MB (adjust)
    UPLOAD_FOLDER_QUIZ = os.path.join('quiz', 'uploads', 'quiz_sources')
