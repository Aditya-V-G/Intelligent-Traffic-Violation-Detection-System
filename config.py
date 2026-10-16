# config.py

import os

class Config:
    """Base configuration class"""

    # Flask Settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'default_secret_key')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'

    # Server Config
    SERVER_HOST = os.getenv('SERVER_HOST', 'localhost')
    SERVER_PORT = int(os.getenv('SERVER_PORT', 5000))

    # Paths
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads/')
    STATIC_FOLDER = os.getenv('STATIC_FOLDER', 'static')
    TEMPLATES_FOLDER = os.getenv('TEMPLATES_FOLDER', 'templates')

    # Logs
    LOG_FOLDER = os.getenv('LOG_FOLDER', 'logs/')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # CCTV Inputs (optional)
    LIVE_CCTV_IPS = os.getenv('LIVE_CCTV_IPS', '').split(',')
    LIVE_CCTV_FILES = os.getenv('LIVE_CCTV_FILES', '').split(',')

    # MySQL Database URL (Not used directly because you use pymysql)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL',
        'mysql+mysqlconnector://root:root@localhost/mydb'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
