# app_instance.py

import os
from flask import Flask
from config import Config

def create_app():
    """
    Create and configure the Flask app instance.
    """
    app = Flask(
        __name__,
        template_folder=Config.TEMPLATES_FOLDER,
        static_folder=Config.STATIC_FOLDER
    )

    # Load configuration
    app.config.from_object(Config)

    # Ensure Upload folder exists
    upload_folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Ensure Logs folder exists
    log_folder = app.config['LOG_FOLDER']
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    print(f"✔ Flask App Initialized")
    print(f"✔ Uploads: {upload_folder}")
    print(f"✔ Logs: {log_folder}")

    return app
