# wsgi.py
import sys
import os
import traceback

print("🚀 WSGI: Starting application...", file=sys.stderr)
print(f"🚀 WSGI: Current directory: {os.getcwd()}", file=sys.stderr)
print(f"🚀 WSGI: Python path: {sys.path}", file=sys.stderr)

try:
    print("🚀 WSGI: Attempting to import app...", file=sys.stderr)
    from app import app, socketio
    print("✅ WSGI: Successfully imported app and socketio", file=sys.stderr)
    
    # Test database connection
    try:
        from app import get_db_connection
        conn = get_db_connection()
        if conn:
            print("✅ WSGI: Database connection successful", file=sys.stderr)
            conn.close()
        else:
            print("❌ WSGI: Database connection failed", file=sys.stderr)
    except Exception as e:
        print(f"❌ WSGI: Database connection error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    
    application = app
    print("✅ WSGI: Application ready, will bind to port", file=sys.stderr)
    
except Exception as e:
    print(f"❌ WSGI: Failed to import app: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    raise

if __name__ == "__main__":
    socketio.run(app)