# wsgi.py - Entry point for Gunicorn
from app import app, socketio

# Ensure TensorFlow is configured before anything else
print("🚀 Starting application via WSGI...")

# This is the standard way to expose the application for Gunicorn
# when using Flask-SocketIO
application = app

if __name__ == "__main__":
    socketio.run(app)