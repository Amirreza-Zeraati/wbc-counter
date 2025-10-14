import os


class Config:
    SECRET_KEY = 'cats'
    UPLOAD_FOLDER = 'static/uploads'
    RESULTS_FOLDER = 'static/uploads/results'
    MAX_FILE_SIZE = 16 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

    @staticmethod
    def init_app(app):
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.RESULTS_FOLDER, exist_ok=True)
