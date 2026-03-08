from flask import Flask, render_template, request, jsonify, send_file, Response, send_from_directory, redirect
import pandas as pd
import os
import pg8000
import json
from math import radians, sin, cos, sqrt, atan2, degrees
from datetime import datetime
import heapq
from collections import defaultdict
import traceback
import math
from werkzeug.utils import secure_filename
import glob
import time
import threading
from flask_socketio import SocketIO, emit, join_room, leave_room
import hashlib
import secrets
from nlp_processor import create_nlp_processor
from advanced_nlp_processor import create_advanced_nlp_processor
from voice_processor import get_voice_processor, voice_to_text
import tensorflow as tf
from feedback_module import feedback_bp
from functools import wraps
import jwt  # PyJWT library
from datetime import datetime, timedelta

print("🚀 DEBUG: Starting app.py initialization...")

# ============= RENDER.COM COMPATIBILITY ADDITIONS =============
# Import for Render database connection
import sys
print("🚀 DEBUG: App script started", file=sys.stderr)
from dotenv import load_dotenv

# ============= DEBUG STARTUP =============
import sys
print("🚀 DEBUG: App script started", file=sys.stderr)
# =========================================

# Load environment variables from .env file (if it exists)
load_dotenv()

# Determine if we're running on Render
IS_RENDER = os.environ.get('RENDER', False) or 'RENDER' in os.environ

# Monkey patch for eventlet when running on Render with gunicorn
if not __name__ == '__main__':
    try:
        import eventlet
        eventlet.monkey_patch()
        print("✓ Eventlet monkey patch applied for Render")
    except ImportError:
        print("⚠ Eventlet not available, WebSockets may not work properly")

# ============= END RENDER.COM ADDITIONS =============

# ---------------------------------------------------
# GLOBAL NLP PROCESSOR
# ---------------------------------------------------
nlp_processor = None

print("🔧 DEBUG: Creating Flask app...")
app = Flask(__name__)
print("🔑 DEBUG: Setting secret key...")
app.config['SECRET_KEY'] = secrets.token_hex(16)

# IMPORTANT: Use eventlet for async_mode with gunicorn
# This configuration is critical for Render deployment
print("🔌 DEBUG: Initializing SocketIO with eventlet...")
socketio = SocketIO(app, 
                   cors_allowed_origins="*", 
                   async_mode='eventlet',  # Must be 'eventlet' for gunicorn worker
                   logger=True,  # Enable logging for debugging
                   engineio_logger=True)  # Enable engine.io logging

# Register the feedback blueprint
print("📋 DEBUG: Registering blueprints...")
app.register_blueprint(feedback_bp, url_prefix='/api/feedback')

# ---------------------------------------------------
# AUTHENTICATION SETUP
# ---------------------------------------------------

# Simple user database (in production, use a real database)
USERS = {
    'demo': {
        'password': 'password',
        'name': 'Demo User',
        'role': 'user'
    },
    'admin': {
        'password': 'admin123',
        'name': 'Administrator',
        'role': 'admin'
    }
}

# JWT secret key
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        
        if not token:
            return redirect('/login')
        
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            current_user = USERS.get(data['username'])
            if not current_user:
                return redirect('/login')
        except:
            return redirect('/login')
        
        return f(current_user, *args, **kwargs)
    
    return decorated

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'buildings.csv')
PHOTOS_DIR = os.path.join(BASE_DIR, 'static', 'photos')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
EXPORT_DIR = os.path.join(BASE_DIR, 'exports')

# Create directories if they don't exist
for directory in [PHOTOS_DIR, UPLOAD_DIR, EXPORT_DIR]:
    os.makedirs(directory, exist_ok=True)
# ---------------------------------------------------
# REAL-TIME TRACKING DATA STRUCTURES
# ---------------------------------------------------
active_trackers = {}  # {tracker_id: {user_id, name, latitude, longitude, timestamp, color, icon, mode}}
tracker_connections = {}  # {tracker_id: [socket_ids]}
user_sessions = {}  # {socket_id: {user_id, tracker_id}}
TRACKER_TIMEOUT = 300  # 5 minutes timeout for inactive trackers

# Generate unique colors for trackers
TRACKER_COLORS = [
    '#FF6B6B', '#4ECDC4', '#FFD166', '#9B5DE5', '#00BBF9', 
    '#00F5D4', '#FF97B7', '#8AC926', '#1982C4', '#6A4C93'
]
TRACKER_ICONS = [
    'walking', 'bicycle', 'car', 'motorcycle', 'running',
    'hiking', 'wheelchair', 'shipping-fast', 'taxi', 'bus'
]

# ---------------------------------------------------
# ALLOWED PHOTO EXTENSIONS
# ---------------------------------------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------------------------------
# ICON CONFIGURATION WITH PHOTO SUPPORT
# ---------------------------------------------------
ICON_CONFIG = {
    "academic_block": {"icon": "university", "color": "#3498db", "name": "Academic Block", "photos": []},
    "lecture_hall": {"icon": "chalkboard-teacher", "color": "#2980b9", "name": "Lecture Hall", "photos": []},
    "library": {"icon": "book", "color": "#1f618d", "name": "Library", "photos": []},
    "laboratory": {"icon": "flask", "color": "#5dade2", "name": "Laboratory", "photos": []},
    "computer_lab": {"icon": "desktop", "color": "#2e86c1", "name": "Computer Lab", "photos": []},
    "research_center": {"icon": "microscope", "color": "#2874a6", "name": "Research Center", "photos": []},
    "classroom": {"icon": "chalkboard", "color": "#21618c", "name": "Classroom", "photos": []},
    "administration": {"icon": "landmark", "color": "#8e44ad", "name": "Administration", "photos": []},
    "office": {"icon": "building", "color": "#9b59b6", "name": "Office", "photos": []},
    "hostel": {"icon": "home", "color": "#27ae60", "name": "Hostel", "photos": []},
    "dormitory": {"icon": "house-user", "color": "#229954", "name": "Dormitory", "photos": []},
    "cafeteria": {"icon": "utensils", "color": "#2ecc71", "name": "Cafeteria", "photos": []},
    "canteen": {"icon": "hamburger", "color": "#28b463", "name": "Canteen", "photos": []},
    "restaurant": {"icon": "utensils", "color": "#e74c3c", "name": "Restaurant", "photos": []},
    "cafe": {"icon": "coffee", "color": "#8b4513", "name": "Cafe", "photos": []},
    "sports_facility": {"icon": "futbol", "color": "#e67e22", "name": "Sports Facility", "photos": []},
    "gymnasium": {"icon": "dumbbell", "color": "#e74c3c", "name": "Gymnasium", "photos": []},
    "stadium": {"icon": "baseball-ball", "color": "#cb4335", "name": "Stadium", "photos": []},
    "swimming_pool": {"icon": "swimming-pool", "color": "#138d75", "name": "Swimming Pool", "photos": []},
    "gate": {"icon": "door-open", "color": "#e74c3c", "name": "Gate", "photos": []},
    "main_gate": {"icon": "monument", "color": "#dc7633", "name": "Main Gate", "photos": []},
    "parking": {"icon": "parking", "color": "#95a5a6", "name": "Parking", "photos": []},
    "bus_stop": {"icon": "bus", "color": "#f39c12", "name": "Bus Stop", "photos": []},
    "taxi_stand": {"icon": "taxi", "color": "#f1c40f", "name": "Taxi Stand", "photos": []},
    "clinic": {"icon": "clinic-medical", "color": "#e74c3c", "name": "Clinic", "photos": []},
    "hospital": {"icon": "hospital", "color": "#c0392b", "name": "Hospital", "photos": []},
    "pharmacy": {"icon": "pills", "color": "#e74c3c", "name": "Pharmacy", "photos": []},
    "workshop": {"icon": "tools", "color": "#34495e", "name": "Workshop", "photos": []},
    "auditorium": {"icon": "theater-masks", "color": "#5d6d7e", "name": "Auditorium", "photos": []},
    "conference_hall": {"icon": "comments", "color": "#4a235a", "name": "Conference Hall", "photos": []},
    "washroom": {"icon": "restroom", "color": "#a569bd", "name": "Washroom", "photos": []},
    "store": {"icon": "warehouse", "color": "#dc7633", "name": "Store", "photos": []},
    "shop": {"icon": "shopping-cart", "color": "#f1948a", "name": "Shop", "photos": []},
    "atm": {"icon": "money-bill-wave", "color": "#27ae60", "name": "ATM", "photos": []},
    "bank": {"icon": "university", "color": "#229954", "name": "Bank", "photos": []},
    "garden": {"icon": "seedling", "color": "#52be80", "name": "Garden", "photos": []},
    "park": {"icon": "tree", "color": "#239b56", "name": "Park", "photos": []},
    "chapel": {"icon": "pray", "color": "#9b59b6", "name": "Chapel", "photos": []},
    "mosque": {"icon": "mosque", "color": "#2e86c1", "name": "Mosque", "photos": []},
    "security_post": {"icon": "shield-alt", "color": "#2c3e50", "name": "Security Post", "photos": []},
    "police_post": {"icon": "police-box", "color": "#21618c", "name": "Police Post", "photos": []},
    "music_room": {"icon": "music", "color": "#8e44ad", "name": "Music Room", "photos": []},
    "art_gallery": {"icon": "palette", "color": "#af7ac5", "name": "Art Gallery", "photos": []},
    "theater": {"icon": "theater-masks", "color": "#6c3483", "name": "Theater", "photos": []},
    "student_center": {"icon": "users", "color": "#229954", "name": "Student Center", "photos": []},
    "reception": {"icon": "headset", "color": "#7d3c98", "name": "Reception", "photos": []},
    "registrar": {"icon": "file-contract", "color": "#6c3483", "name": "Registrar", "photos": []},
    "finance": {"icon": "money-check", "color": "#884ea0", "name": "Finance Office", "photos": []},
    "default": {"icon": "map-marker-alt", "color": "#7d3c98", "name": "Building", "photos": []}
}

# ---------------------------------------------------
# SPEED SETTINGS (km/h)
# ---------------------------------------------------
SPEEDS = {
    'walking': 4.0,
    'cycling': 15.0,
    'driving': 20.0
}

# ---------------------------------------------------
# ROUTE COLORS
# ---------------------------------------------------
ROUTE_COLORS = {
    'walking': '#FF6B6B',
    'cycling': '#4ECDC4',
    'driving': '#FFD166',
    'default': '#9B5DE5'
}

# ---------------------------------------------------
# PATH TYPE COMPATIBILITY
# ---------------------------------------------------
PATH_COMPATIBILITY = {
    'walking': ['walking', 'cycling', 'driving'],
    'cycling': ['cycling', 'driving'],
    'driving': ['driving']
}

# ---------------------------------------------------
# MAP TILE CONFIGURATION
# ---------------------------------------------------
MAP_TILES = {
    'osm_standard': {
        'name': 'OpenStreetMap Standard',
        'url': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap contributors',
        'default': True
    },
    'osm_hot': {
        'name': 'OpenStreetMap HOT',
        'url': 'https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap contributors, Tiles style by Humanitarian OpenStreetMap Team',
        'default': False
    },
    'cartodb_positron': {
        'name': 'CartoDB Positron',
        'url': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        'attribution': '© OpenStreetMap contributors, © CARTO',
        'default': False
    },
    'cartodb_dark': {
        'name': 'CartoDB Dark',
        'url': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        'attribution': '© OpenStreetMap contributors, © CARTO',
        'default': False
    },
    'esri_world_imagery': {
        'name': 'Esri World Imagery',
        'url': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'attribution': 'Tiles © Esri — Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community',
        'default': False
    },
    'stamen_toner': {
        'name': 'Stamen Toner',
        'url': 'https://stamen-tiles-{s}.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        'attribution': 'Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        'default': False
    },
    'stamen_watercolor': {
        'name': 'Stamen Watercolor',
        'url': 'https://stamen-tiles-{s}.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
        'attribution': 'Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors',
        'default': False
    },
    'wikimedia': {
        'name': 'Wikimedia Maps',
        'url': 'https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap contributors',
        'default': False
    },
    'opentopomap': {
        'name': 'OpenTopoMap',
        'url': 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        'attribution': 'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)',
        'default': False
    }
}

# ---------------------------------------------------
# LOAD BUILDINGS WITH PHOTO SUPPORT
# ---------------------------------------------------
def load_buildings_with_photos():
    try:
        df = pd.read_csv(CSV_PATH, sep='\t')
        print(f"✓ CSV loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        df = pd.DataFrame(columns=['name', 'type', 'latitude', 'longitude'])

    # Validate required columns
    required_cols = {"name", "type", "latitude", "longitude"}
    if not required_cols.issubset(set(df.columns)):
        print(f"⚠ Warning: CSV missing required columns. Found: {set(df.columns)}")
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        df['latitude'] = -0.5075
        df['longitude'] = 37.4575
    
    # Clean data
    df = df.dropna(subset=['name', 'latitude', 'longitude'])
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])

    # Add ID column
    df['id'] = range(1, len(df) + 1)
    
    # Add enhanced description
    def create_description(row):
        building_type = str(row.get('type', 'default'))
        config = ICON_CONFIG.get(building_type, ICON_CONFIG['default'])
        building_type_name = config.get('name', building_type.replace('_', ' ').title())
        
        descriptions = {
            'academic_block': f"Academic building for lectures and classes at University of Embu",
            'lecture_hall': f"Large lecture hall for classroom sessions",
            'library': f"University library with study resources and books",
            'laboratory': f"Science laboratory for practical experiments",
            'computer_lab': f"Computer laboratory with workstations",
            'research_center': f"Research center for academic studies",
            'classroom': f"Standard classroom for teaching",
            'administration': f"Administrative offices of University of Embu",
            'office': f"Staff and faculty offices",
            'hostel': f"Student accommodation with rooms",
            'dormitory': f"Student dormitory building",
            'cafeteria': f"Dining facility for students and staff",
            'canteen': f"Food service area",
            'restaurant': f"Full-service restaurant",
            'cafe': f"Coffee shop and snacks",
            'sports_facility': f"Sports and fitness facility",
            'gymnasium': f"Indoor gym with exercise equipment",
            'stadium': f"Sports stadium for events",
            'swimming_pool': f"Swimming pool facility",
            'gate': f"Campus entrance/exit point",
            'main_gate': f"Main campus entrance",
            'parking': f"Vehicle parking area",
            'bus_stop': f"Bus pick-up and drop-off point",
            'taxi_stand': f"Taxi waiting area",
            'clinic': f"Health clinic for medical services",
            'hospital': f"Medical hospital facility",
            'pharmacy': f"Medicine and drug store",
            'workshop': f"Technical workshop for practical training",
            'auditorium': f"Large event venue for performances",
            'conference_hall': f"Conference and meeting facility",
            'washroom': f"Public restroom facilities",
            'store': f"Storage and supplies facility",
            'shop': f"Retail shop",
            'atm': f"Automated Teller Machine",
            'bank': f"Banking facility",
            'garden': f"Garden with plants and flowers",
            'park': f"Park area with greenery",
            'chapel': f"Religious chapel for worship",
            'mosque': f"Islamic prayer center",
            'security_post': f"Campus security office",
            'police_post': f"Police station",
            'music_room': f"Music practice room",
            'art_gallery': f"Art exhibition space",
            'theater': f"Performance theater",
            'student_center': f"Student activities and services center",
            'reception': f"Main reception area",
            'registrar': f"Student registration and records office",
            'finance': f"Financial services and bursar office"
        }
        
        return descriptions.get(building_type, f"{row['name']} - {building_type_name} facility at University of Embu")

    df['description'] = df.apply(create_description, axis=1)

    # Add category
    building_categories = {
        'Academic': ['academic_block', 'lecture_hall', 'library', 'laboratory', 'computer_lab', 'research_center', 'classroom'],
        'Administrative': ['administration', 'office', 'reception', 'registrar', 'finance'],
        'Residential': ['hostel', 'dormitory'],
        'Dining': ['cafeteria', 'canteen', 'restaurant', 'cafe'],
        'Sports & Recreation': ['sports_facility', 'gymnasium', 'stadium', 'swimming_pool'],
        'Infrastructure': ['gate', 'main_gate', 'parking', 'bus_stop', 'taxi_stand'],
        'Health & Wellness': ['clinic', 'hospital', 'pharmacy'],
        'Technical': ['workshop', 'auditorium', 'conference_hall'],
        'Services': ['washroom', 'store', 'shop', 'atm', 'bank'],
        'Natural Areas': ['garden', 'park'],
        'Cultural & Religious': ['chapel', 'mosque'],
        'Security': ['security_post', 'police_post'],
        'Arts & Music': ['music_room', 'art_gallery', 'theater'],
        'Student Life': ['student_center'],
        'Other': ['default']
    }

    def get_category(building_type):
        building_type = str(building_type)
        for category, types in building_categories.items():
            if building_type in types:
                return category
        return 'Other'

    df['category'] = df['type'].apply(get_category)

    # Load photos for each building
    def get_building_photos(building_id, building_name):
        photos = []
        building_name_clean = building_name.lower().replace(' ', '_').replace('.', '').replace(',', '')
        
        patterns = [
            f"{building_id}_{building_name_clean}*.jpg",
            f"{building_id}_{building_name_clean}*.jpeg",
            f"{building_id}_{building_name_clean}*.png",
            f"{building_id}_{building_name_clean}*.webp",
            f"{building_name_clean}*.jpg",
            f"{building_name_clean}*.jpeg",
            f"{building_name_clean}*.png",
            f"{building_name_clean}*.webp",
            f"{building_id}_*.jpg",
            f"{building_id}_*.jpeg",
            f"{building_id}_*.png",
            f"{building_id}_*.webp",
            f"building_{building_id}*.jpg",
            f"building_{building_id}*.jpeg",
            f"building_{building_id}*.png",
            f"building_{building_id}*.webp"
        ]
        
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(PHOTOS_DIR, pattern)):
                filename = os.path.basename(filepath)
                if filename not in [p['filename'] for p in photos]:
                    photos.append({
                        'filename': filename,
                        'url': f'/static/photos/{filename}',
                        'title': building_name
                    })
        
        # If no specific photos found, use type-based default photos
        if not photos:
            building_type = df[df['id'] == building_id]['type'].iloc[0] if building_id in df['id'].values else 'default'
            type_photos = ICON_CONFIG.get(building_type, ICON_CONFIG['default']).get('photos', [])
            photos.extend(type_photos)
        
        return photos

    # Create a list of building photos
    building_photos = {}
    for _, row in df.iterrows():
        photos = get_building_photos(int(row['id']), str(row['name']))
        building_photos[int(row['id'])] = photos
    
    print(f"✓ Total buildings loaded: {len(df)}")
    print(f"  Categories: {df['category'].unique().tolist()}")
    print(f"  Photos found: {sum(len(photos) for photos in building_photos.values())} total photos")
    
    return df, building_photos

# ---------------------------------------------------
# SAMPLE PHOTOS CREATION
# ---------------------------------------------------
def create_sample_photos(df, building_photos):
    """Create sample photos for testing"""
    try:
        print("🎨 Creating sample photos for buildings...")
        
        # Try to import PIL
        try:
            from PIL import Image, ImageDraw, ImageFont
            import textwrap
        except ImportError:
            print("⚠ PIL/Pillow not installed. Install with: pip install pillow")
            return
        
        # Create a simple placeholder image
        def create_placeholder_image(building_name, building_type, output_path):
            img = Image.new('RGB', (400, 300), color=(73, 109, 137))
            d = ImageDraw.Draw(img)
            
            # Add building name (wrapped)
            font = ImageFont.load_default()
            wrapped_text = textwrap.fill(building_name, width=20)
            d.text((10, 10), wrapped_text, fill=(255, 255, 255), font=font)
            
            # Add building type
            d.text((10, 100), f"Type: {building_type}", fill=(255, 255, 200), font=font)
            
            # Add "Sample Photo" text
            d.text((10, 150), "Sample Photo - Click to upload real photos", 
                   fill=(200, 255, 200), font=font)
            
            # Add decorative elements
            d.rectangle([5, 5, 395, 295], outline=(255, 255, 255), width=2)
            
            img.save(output_path)
            print(f"  Created placeholder: {output_path}")
        
        # Create sample photos for first 5 buildings
        for i, (_, building) in enumerate(df.head(5).iterrows()):
            building_id = int(building['id'])
            building_name = str(building['name'])
            building_type = str(building['type'])
            
            # Create photo filename
            filename = f"{building_id}_{building_name.lower().replace(' ', '_')}_sample.jpg"
            filepath = os.path.join(PHOTOS_DIR, filename)
            
            # Skip if already exists
            if os.path.exists(filepath):
                continue
            
            # Create placeholder image
            try:
                create_placeholder_image(building_name, building_type, filepath)
                
                # Add to building_photos dictionary
                if building_id not in building_photos:
                    building_photos[building_id] = []
                
                # Check if this photo already exists in the list
                if not any(p['filename'] == filename for p in building_photos[building_id]):
                    building_photos[building_id].append({
                        'filename': filename,
                        'url': f'/static/photos/{filename}',
                        'title': f"{building_name} - Sample",
                        'is_sample': True
                    })
            except Exception as e:
                print(f"  ⚠ Error creating sample photo for {building_name}: {e}")
                continue
        
        print("✅ Sample photos created!")
        
    except Exception as e:
        print(f"⚠ Error in create_sample_photos: {e}")
        print("  Sample photos not created. Continuing without them...")

# After loading buildings
print("📊 DEBUG: Loading buildings with photos...", file=sys.stderr)
df, building_photos = load_buildings_with_photos()
print(f"✅ DEBUG: Loaded {len(df)} buildings", file=sys.stderr)

# After creating sample photos
print("🖼️ DEBUG: Creating sample photos...", file=sys.stderr)
create_sample_photos(df, building_photos)
print("✅ DEBUG: Sample photos created", file=sys.stderr)

# Before NLP processor initialization
print("🧠 DEBUG: Initializing NLP processor...", file=sys.stderr)
if nlp_processor is None:
    building_names = df['name'].tolist()
    nlp_processor = create_nlp_processor(building_names, use_advanced=False)
print(f"✅ DEBUG: NLP processor ready. Type: {type(nlp_processor).__name__}", file=sys.stderr)

# ---------------------------------------------------
# VERIFY NLP PROCESSOR
# ---------------------------------------------------
if nlp_processor is None:
    print("⚠ Warning: NLP processor not initialized, re-initializing...")
    building_names = df['name'].tolist()
    nlp_processor = create_nlp_processor(building_names, use_advanced=False)
    print("✅ NLP processor re-initialized")

print(f"🤖 NLP Processor Type: {type(nlp_processor).__name__}")

# ---------------------------------------------------
# DATABASE CONNECTION - MODIFIED FOR RENDER
# ---------------------------------------------------
def get_db_connection():
    """Get database connection - works on both local and Render"""
    try:
        # Check if we're on Render and DATABASE_URL is provided
        database_url = os.environ.get('DATABASE_URL')
        
        if database_url:
            # Render provides DATABASE_URL automatically
            # Convert postgres:// to postgresql:// if needed (psycopg2 requirement)
            if database_url.startswith('postgres://'):
                database_url = database_url.replace('postgres://', 'postgresql://', 1)
            
            # Use psycopg2 for Render (better compatibility)
            import psycopg2
            conn = psycopg2.connect(database_url)
            print("✓ Connected to Render PostgreSQL database")
            return conn
        else:
            # Local development with pg8000
            conn = pg8000.connect(
                user=os.environ.get('DB_USER', 'postgres'),
                password=os.environ.get('DB_PASSWORD', '37193083591'),
                host=os.environ.get('DB_HOST', 'localhost'),
                port=int(os.environ.get('DB_PORT', '5432')),
                database=os.environ.get('DB_NAME', 'embu_navigation')
            )
            print("✓ Connected to local PostgreSQL database")
            return conn
            
    except ImportError:
        # Fallback if psycopg2 not available
        try:
            conn = pg8000.connect(
                user=os.environ.get('DB_USER', 'postgres'),
                password=os.environ.get('DB_PASSWORD', '37193083591'),
                host=os.environ.get('DB_HOST', 'localhost'),
                port=int(os.environ.get('DB_PORT', '5432')),
                database=os.environ.get('DB_NAME', 'embu_navigation')
            )
            return conn
        except Exception as e:
            print(f"✗ Database connection error: {e}")
            return None
    except Exception as e:
        print(f"✗ Database connection error: {e}")
        return None

# After database connection function definition (optional)
print("💾 DEBUG: Database connection function defined", file=sys.stderr)

# ---------------------------------------------------
# LOAD PATHS FROM POSTGIS WITH MODE FILTERING
# ---------------------------------------------------
def get_paths(mode=None):
    """Get paths filtered by transport mode"""
    try:
        conn = get_db_connection()
        if conn is None:
            print("⚠ Database connection failed, returning sample paths")
            return get_sample_paths(mode)
        
        cursor = conn.cursor()
        
        # Check if path_type column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='embu_paths' AND column_name='path_type';
        """)
        has_path_type = cursor.fetchone() is not None
        
        if mode and has_path_type:
            # Filter paths based on mode compatibility
            if mode == 'walking':
                query = """
                    SELECT id, ST_AsGeoJSON(geom), path_type, 
                           ST_Length(geom::geography) as length,
                           name
                    FROM embu_paths 
                    WHERE path_type IN ('walking', 'cycling', 'driving')
                """
            elif mode == 'cycling':
                query = """
                    SELECT id, ST_AsGeoJSON(geom), path_type, 
                           ST_Length(geom::geography) as length,
                           name
                    FROM embu_paths 
                    WHERE path_type IN ('cycling', 'driving')
                """
            elif mode == 'driving':
                query = """
                    SELECT id, ST_AsGeoJSON(geom), path_type, 
                           ST_Length(geom::geography) as length,
                           name
                    FROM embu_paths 
                    WHERE path_type = 'driving'
                """
        elif has_path_type:
            # Get all paths with their types
            query = """
                SELECT id, ST_AsGeoJSON(geom), path_type, 
                       ST_Length(geom::geography) as length,
                       name
                FROM embu_paths
            """
        else:
            # Legacy table without path_type
            query = """
                SELECT id, ST_AsGeoJSON(geom), 'walking' as path_type, 
                       ST_Length(geom::geography) as length,
                       '' as name
                FROM embu_paths;
            """
        
        cursor.execute(query)
        paths = cursor.fetchall()
        cursor.close()
        conn.close()
        
        print(f"✓ Loaded {len(paths)} path segments for mode: {mode or 'all'}")
        return paths
    except Exception as e:
        print(f"✗ Error loading paths: {e}")
        traceback.print_exc()
        return get_sample_paths(mode)

def get_sample_paths(mode=None):
    """Return sample paths when database is unavailable"""
    print(f"⚠ Using sample paths for mode: {mode or 'all'}")
    
    # Create sample paths around campus center
    center_lat = -0.5075
    center_lon = 37.4575
    
    sample_paths = []
    path_types = ['walking', 'cycling', 'driving']
    
    # Create a grid of paths with different types
    path_id = 1
    for i in range(-3, 4):
        for j in range(-3, 4):
            # Determine path type based on position
            if i % 3 == 0:  # Every 3rd row as driving
                path_type = 'driving'
            elif i % 2 == 0:  # Every 2nd row as cycling
                path_type = 'cycling'
            else:  # Others as walking
                path_type = 'walking'
            
            # Skip if not compatible with requested mode
            if mode:
                if mode == 'driving' and path_type != 'driving':
                    continue
                elif mode == 'cycling' and path_type not in ['cycling', 'driving']:
                    continue
            
            # Horizontal path
            start_lat = center_lat + (i * 0.0003)
            start_lon = center_lon + (j * 0.0003)
            end_lat = start_lat
            end_lon = start_lon + 0.0006
            
            geojson = {
                "type": "LineString",
                "coordinates": [
                    [start_lon, start_lat],
                    [end_lon, end_lat]
                ]
            }
            
            sample_paths.append((
                path_id,
                json.dumps(geojson),
                path_type,
                66.7,  # Approximate length
                f"{path_type} path {path_id}"
            ))
            path_id += 1
    
    print(f"Created {len(sample_paths)} sample paths")
    return sample_paths

# ---------------------------------------------------
# HAVERSINE DISTANCE
# ---------------------------------------------------
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371000
    
    lat1_rad = radians(float(lat1))
    lon1_rad = radians(float(lon1))
    lat2_rad = radians(float(lat2))
    lon2_rad = radians(float(lon2))
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

# ---------------------------------------------------
# IMPROVED PATH FINDER WITH MODE SUPPORT
# ---------------------------------------------------
class SimplePathFinder:
    def __init__(self, paths_data, mode='walking'):
        self.paths = []
        self.path_segments = []
        self.nodes = set()
        self.graph = defaultdict(list)
        self.mode = mode
        
        if paths_data:
            self._parse_paths(paths_data)
            self._build_graph()
    
    def _parse_paths(self, paths_data):
        """Parse paths from database into segments and nodes"""
        print(f"🔍 Parsing {len(paths_data)} paths for {self.mode} mode...")
        
        segment_count = 0
        node_count = 0
        
        for path_id, geojson_str, path_type, length, name in paths_data:
            try:
                # Check if path type is compatible with current mode
                if not self._is_path_compatible(path_type):
                    continue
                
                # Parse GeoJSON
                if isinstance(geojson_str, str):
                    geojson = json.loads(geojson_str)
                else:
                    geojson = geojson_str
                
                if geojson['type'] == 'LineString' and 'coordinates' in geojson:
                    coords = geojson['coordinates']
                    
                    if len(coords) >= 2:
                        # Convert to (lat, lon) format from [lon, lat]
                        points = [(coord[1], coord[0]) for coord in coords]
                        
                        # Store the complete path
                        self.paths.append({
                            'id': path_id,
                            'points': points,
                            'type': path_type,
                            'name': name or f"Path {path_id}",
                            'length': float(length) if length else 0
                        })
                        
                        # Break into segments
                        for i in range(len(points) - 1):
                            start_point = points[i]
                            end_point = points[i + 1]
                            
                            # Round points for node matching
                            start_node = (round(start_point[0], 6), round(start_point[1], 6))
                            end_node = (round(end_point[0], 6), round(end_point[1], 6))
                            
                            segment = {
                                'id': f"{path_id}_{i}",
                                'start': start_point,
                                'end': end_point,
                                'start_node': start_node,
                                'end_node': end_node,
                                'points': [start_point, end_point],
                                'type': path_type,
                                'name': name or f"Path {path_id}",
                                'length': haversine_distance(
                                    start_point[1], start_point[0],
                                    end_point[1], end_point[0]
                                )
                            }
                            self.path_segments.append(segment)
                            
                            # Add nodes
                            self.nodes.add(start_node)
                            self.nodes.add(end_node)
                            
                            segment_count += 1
                            node_count = len(self.nodes)
                    
                elif geojson['type'] == 'MultiLineString' and 'coordinates' in geojson:
                    # Handle MultiLineString
                    for line in geojson['coordinates']:
                        if len(line) >= 2:
                            points = [(coord[1], coord[0]) for coord in line]
                            
                            for i in range(len(points) - 1):
                                start_point = points[i]
                                end_point = points[i + 1]
                                
                                # Round points for node matching
                                start_node = (round(start_point[0], 6), round(start_point[1], 6))
                                end_node = (round(end_point[0], 6), round(end_point[1], 6))
                                
                                segment = {
                                    'id': f"{path_id}_ml_{i}",
                                    'start': start_point,
                                    'end': end_point,
                                    'start_node': start_node,
                                    'end_node': end_node,
                                    'points': [start_point, end_point],
                                    'type': path_type,
                                    'name': name or f"Path {path_id}",
                                    'length': haversine_distance(
                                        start_point[1], start_point[0],
                                        end_point[1], end_point[0]
                                    )
                                }
                                self.path_segments.append(segment)
                                
                                # Add nodes
                                self.nodes.add(start_node)
                                self.nodes.add(end_node)
                                
                                segment_count += 1
                                node_count = len(self.nodes)
                
            except Exception as e:
                print(f"⚠ Error parsing path {path_id}: {e}")
                continue
        
        print(f"✓ Parsed {segment_count} segments, {node_count} nodes for {self.mode} mode")
    
    def _is_path_compatible(self, path_type):
        """Check if path type is compatible with current mode"""
        return path_type in PATH_COMPATIBILITY.get(self.mode, ['walking', 'cycling', 'driving'])
    
    def _build_graph(self):
        """Build graph from segments"""
        print(f"🔄 Building graph from {len(self.path_segments)} segments...")
        
        for segment in self.path_segments:
            start_node = segment['start_node']
            end_node = segment['end_node']
            weight = segment['length']
            path_type = segment['type']
            
            # Add bidirectional edges
            self.graph[start_node].append((end_node, weight, segment['id'], path_type))
            self.graph[end_node].append((start_node, weight, segment['id'], path_type))
        
        print(f"✓ Graph built with {len(self.graph)} nodes for {self.mode} mode")
    
    def find_nearest_node(self, lat, lon, max_distance=200):
        """Find the nearest graph node to the given coordinates"""
        nearest_node = None
        nearest_distance = float('inf')
        
        for node in self.graph.keys():
            node_lat, node_lon = node
            distance = haversine_distance(lon, lat, node_lon, node_lat)
            
            if distance < nearest_distance and distance < max_distance:
                nearest_distance = distance
                nearest_node = node
        
        return nearest_node, nearest_distance
    
    def find_route(self, start_lat, start_lon, end_lat, end_lon):
        """Find route using Dijkstra's algorithm with better error handling"""
        try:
            if not self.graph or len(self.graph) == 0:
                print("⚠ No graph available for pathfinding - using direct route")
                return self._get_direct_route(start_lat, start_lon, end_lat, end_lon)
            
            print(f"  Finding {self.mode} route: ({start_lat:.6f}, {start_lon:.6f}) -> ({end_lat:.6f}, {end_lon:.6f})")
            
            # Find nearest nodes with expanded search radius
            start_node, start_dist = self.find_nearest_node(start_lat, start_lon, max_distance=200)
            end_node, end_dist = self.find_nearest_node(end_lat, end_lon, max_distance=200)
            
            if not start_node:
                print(f"⚠ Could not find start node near ({start_lat}, {start_lon})")
                return self._get_direct_route(start_lat, start_lon, end_lat, end_lon)
            
            if not end_node:
                print(f"⚠ Could not find end node near ({end_lat}, {end_lon})")
                return self._get_direct_route(start_lat, start_lon, end_lat, end_lon)
            
            print(f"  Start node: {start_node} (distance: {start_dist:.1f}m)")
            print(f"  End node: {end_node} (distance: {end_dist:.1f}m)")
            
            # Dijkstra's algorithm
            distances = {node: float('inf') for node in self.graph}
            distances[start_node] = 0
            previous = {node: None for node in self.graph}
            previous_segment = {node: None for node in self.graph}
            previous_segment_type = {node: None for node in self.graph}
            
            pq = [(0, start_node)]
            
            while pq:
                current_dist, current_node = heapq.heappop(pq)
                
                if current_dist > distances[current_node]:
                    continue
                    
                if current_node == end_node:
                    break
                
                for neighbor, weight, segment_id, segment_type in self.graph[current_node]:
                    new_dist = current_dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous[neighbor] = current_node
                        previous_segment[neighbor] = segment_id
                        previous_segment_type[neighbor] = segment_type
                        heapq.heappush(pq, (new_dist, neighbor))
            
            if distances[end_node] == float('inf'):
                print(f"  ⚠ No {self.mode} path found through graph - using direct route")
                return self._get_direct_route(start_lat, start_lon, end_lat, end_lon)
            
            # Reconstruct path
            path_nodes = []
            path_segments = []
            path_segment_types = []
            current = end_node
            
            while current is not None:
                path_nodes.append(current)
                if previous[current] is not None:
                    path_segments.append(previous_segment[current])
                    path_segment_types.append(previous_segment_type[current])
                current = previous[current]
            
            path_nodes.reverse()
            path_segments.reverse()
            path_segment_types.reverse()
            
            print(f"  ✓ {self.mode} path found: {len(path_nodes)} nodes, distance: {distances[end_node]:.1f}m")
            
            # Build coordinate path
            coordinate_path = []
            
            # Add start point
            coordinate_path.append([start_lat, start_lon])
            
            # Add connection to first node
            if start_dist > 1:
                start_node_lat, start_node_lon = start_node
                coordinate_path.append([start_node_lat, start_node_lon])
            
            # Add all path nodes
            for node in path_nodes:
                node_lat, node_lon = node
                coordinate_path.append([node_lat, node_lon])
            
            # Add connection to end point
            if end_dist > 1:
                end_node_lat, end_node_lon = end_node
                coordinate_path.append([end_node_lat, end_node_lon])
            
            # Add end point
            coordinate_path.append([end_lat, end_lon])
            
            # Remove consecutive duplicates
            clean_path = []
            for i, point in enumerate(coordinate_path):
                if i == 0 or point != coordinate_path[i-1]:
                    clean_path.append(point)
            
            # Calculate total path distance
            total_distance = distances[end_node] + start_dist + end_dist
            
            print(f"  Clean path: {len(clean_path)} points, total distance: {total_distance:.1f}m")
            
            # Get path composition (what types of paths were used)
            path_composition = {}
            for seg_type in path_segment_types:
                path_composition[seg_type] = path_composition.get(seg_type, 0) + 1
            
            return clean_path, total_distance, path_segment_types, path_composition
            
        except Exception as e:
            print(f"⚠ Error in pathfinding: {e}")
            return self._get_direct_route(start_lat, start_lon, end_lat, end_lon)

    def _get_direct_route(self, start_lat, start_lon, end_lat, end_lon):
        """Return a direct straight-line route as fallback"""
        print("  Using direct route fallback")
        
        # Generate 20 points along straight line for smoother path
        points = []
        steps = 20
        for i in range(steps + 1):
            ratio = i / steps
            lat = start_lat + (end_lat - start_lat) * ratio
            lon = start_lon + (end_lon - start_lon) * ratio
            points.append([lat, lon])
        
        # Calculate distance
        distance = haversine_distance(start_lon, start_lat, end_lon, end_lat)
        
        # Return in the same format as the main method
        return points, distance, ['direct'], {'direct': 1}

# ---------------------------------------------------
# CALCULATE TRAVEL TIME
# ---------------------------------------------------
def calculate_travel_time(distance_meters, mode='walking'):
    speed_kmh = SPEEDS.get(mode, 4.0)
    speed_ms = speed_kmh * 1000 / 3600
    time_seconds = distance_meters / speed_ms
    time_minutes = time_seconds / 60
    return round(time_minutes, 1)

# ---------------------------------------------------
# ROUTE API WITH MODE-SPECIFIC PATHFINDING
# ---------------------------------------------------
@app.route('/health')
def health_check():
    """Simple health check endpoint for keep-alive services"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'UoEM AI Navigation'
    })

@app.route('/route', methods=['GET'])
def route():
    try:
        start_lat = float(request.args.get('start_lat'))
        start_lon = float(request.args.get('start_lon'))
        end_lat = float(request.args.get('end_lat'))
        end_lon = float(request.args.get('end_lon'))
        mode = request.args.get('mode', 'walking')
        
        print(f"\n📍 Routing for {mode}: ({start_lat:.6f}, {start_lon:.6f}) → ({end_lat:.6f}, {end_lon:.6f})")
        
        # Get mode-specific paths from database
        paths_data = get_paths(mode)
        print(f"  Loaded {len(paths_data)} path segments for {mode}")
        
        # Create mode-specific pathfinder
        pathfinder = SimplePathFinder(paths_data, mode)
        
        # Try to find route
        if pathfinder.path_segments:
            route_path, path_distance, path_segment_types, path_composition = pathfinder.find_route(
                start_lat, start_lon, end_lat, end_lon
            )
            
            if route_path and len(route_path) > 1:
                print(f"  ✓ Found {mode} path with {len(route_path)} points")
                print(f"  Path composition: {path_composition}")
                
                # Calculate travel time with mode-specific speeds
                estimated_time = calculate_travel_time(path_distance, mode)
                
                # Calculate times for all modes
                times = {}
                for travel_mode in SPEEDS.keys():
                    times[travel_mode] = calculate_travel_time(path_distance, travel_mode)
                
                # Generate directions
                directions = generate_directions(route_path, path_distance, mode, path_segment_types)
                
                response = {
                    "path": route_path,
                    "distance": round(path_distance, 2),
                    "times": times,
                    "directions": directions,
                    "estimated_time": estimated_time,
                    "success": True,
                    "path_type": "following_paths",
                    "mode": mode,
                    "path_composition": path_composition,
                    "point_count": len(route_path)
                }
                
                return jsonify(response)
        
        # Fallback: direct path
        print(f"  ⚠ Using direct path (fallback)")
        return get_direct_route(start_lat, start_lon, end_lat, end_lon, mode)
        
    except Exception as e:
        print(f"✗ Routing error: {e}")
        traceback.print_exc()
        return jsonify({
            "path": [],
            "distance": 0,
            "times": {"walking": 0, "cycling": 0, "driving": 0},
            "success": False,
            "error": str(e)
        })

def get_direct_route(start_lat, start_lon, end_lat, end_lon, mode):
    """Fallback to direct route when no path is found"""
    # Calculate direct distance
    distance = haversine_distance(start_lon, start_lat, end_lon, end_lat)
    estimated_time = calculate_travel_time(distance, mode)
    
    # Generate simple path
    steps = max(10, int(distance / 10))
    path = calculate_direct_path(start_lat, start_lon, end_lat, end_lon, steps)
    
    # Calculate times for all modes
    times = {}
    for travel_mode in SPEEDS.keys():
        times[travel_mode] = calculate_travel_time(distance, travel_mode)
    
    # Generate basic directions
    directions = [
        {
            'step': 1,
            'instruction': f'Start at your selected location',
            'distance': 0,
            'type': 'start'
        },
        {
            'step': 2,
            'instruction': f'Head directly towards destination',
            'distance': round(distance * 0.5),
            'type': mode
        },
        {
            'step': 3,
            'instruction': 'Continue towards destination',
            'distance': round(distance * 0.8),
            'type': mode
        },
        {
            'step': 4,
            'instruction': 'Arrive at your destination',
            'distance': round(distance),
            'type': 'arrival'
        }
    ]
    
    return jsonify({
        "path": path,
        "distance": round(distance, 2),
        "times": times,
        "directions": directions,
        "estimated_time": estimated_time,
        "success": True,
        "path_type": "direct",
        "mode": mode,
        "point_count": len(path)
    })

def calculate_direct_path(start_lat, start_lon, end_lat, end_lon, steps=10):
    """Generate a simple path between two points"""
    path = []
    
    for i in range(steps + 1):
        ratio = i / steps
        lat = start_lat + (end_lat - start_lat) * ratio
        lon = start_lon + (end_lon - start_lon) * ratio
        
        if 0.3 < ratio < 0.7:
            offset = math.sin(ratio * math.pi) * 0.00005
            lat += offset
            lon += offset
        
        path.append([lat, lon])
    
    return path

def generate_directions(path, total_distance, mode, path_segment_types=None):
    """Generate turn-by-turn directions from coordinate path"""
    if len(path) < 3:
        return [
            {
                'step': 1,
                'instruction': 'Start at your location',
                'distance': 0,
                'type': 'start'
            },
            {
                'step': 2,
                'instruction': f'Head towards destination via {mode} route',
                'distance': round(total_distance * 0.5),
                'type': mode
            },
            {
                'step': 3,
                'instruction': 'Arrive at destination',
                'distance': round(total_distance),
                'type': 'arrival'
            }
        ]
    
    directions = []
    directions.append({
        'step': 1,
        'instruction': 'Start at your location',
        'distance': 0,
        'type': 'start'
    })
    
    step_count = 2
    segment_distance = 0
    
    # Add path type information if available
    if path_segment_types and len(path_segment_types) > 0:
        unique_types = set(path_segment_types)
        if len(unique_types) > 1:
            type_info = "using " + ", ".join([f"{t} paths" for t in unique_types])
            directions.append({
                'step': step_count,
                'instruction': f'Begin {mode} route {type_info}',
                'distance': 0,
                'type': mode
            })
            step_count += 1
    
    for i in range(1, len(path) - 1):
        segment_distance += haversine_distance(
            path[i-1][1], path[i-1][0],
            path[i][1], path[i][0]
        )
        
        if i < len(path) - 1:
            bearing1 = calculate_bearing(path[i-1][0], path[i-1][1], path[i][0], path[i][1])
            bearing2 = calculate_bearing(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            
            turn_angle = abs(bearing2 - bearing1)
            if turn_angle > 180:
                turn_angle = 360 - turn_angle
            
            if turn_angle > 30:
                turn_direction = get_turn_direction(bearing1, bearing2)
                
                # Add path type info if we have it
                path_type_info = ""
                if path_segment_types and i-1 < len(path_segment_types):
                    seg_type = path_segment_types[i-1]
                    if seg_type != mode:
                        path_type_info = f" on {seg_type} path"
                
                directions.append({
                    'step': step_count,
                    'instruction': f'{turn_direction}{path_type_info} and continue',
                    'distance': round(segment_distance),
                    'type': mode
                })
                step_count += 1
                segment_distance = 0
    
    remaining_distance = total_distance - sum(d['distance'] for d in directions if 'distance' in d)
    if remaining_distance > 0:
        directions.append({
            'step': step_count,
            'instruction': 'Continue straight to destination',
            'distance': round(remaining_distance),
            'type': mode
        })
        step_count += 1
    
    directions.append({
        'step': step_count,
        'instruction': 'Arrive at your destination',
        'distance': round(total_distance),
        'type': 'arrival'
    })
    
    return directions

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing between two points in degrees"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - cos(lat1) * cos(lat2) * cos(dlon)
    
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def get_turn_direction(bearing1, bearing2):
    """Determine turn direction based on bearing change"""
    diff = (bearing2 - bearing1 + 360) % 360
    
    if diff > 180:
        turn_type = "left"
    else:
        turn_type = "right"
    
    angle = min(diff, 360 - diff)
    if angle < 45:
        severity = "slight "
    elif angle < 135:
        severity = ""
    else:
        severity = "sharp "
    
    return f"Turn {severity}{turn_type}".strip()

# ---------------------------------------------------
# REFRESH BUILDING PHOTOS FUNCTION
# ---------------------------------------------------
def refresh_building_photos():
    """Refresh building photos from disk"""
    print("🔄 Refreshing building photos...")
    
    for _, row in df.iterrows():
        building_id = int(row['id'])
        building_name = str(row['name'])
        
        photos = []
        building_name_clean = building_name.lower().replace(' ', '_').replace('.', '').replace(',', '')
        
        # Search for photos with various patterns
        patterns = [
            f"{building_id}_{building_name_clean}*.jpg",
            f"{building_id}_{building_name_clean}*.jpeg",
            f"{building_id}_{building_name_clean}*.png",
            f"{building_id}_{building_name_clean}*.webp",
            f"{building_id}_{building_name_clean}*.gif",
            f"{building_name_clean}*.jpg",
            f"{building_name_clean}*.jpeg",
            f"{building_name_clean}*.png",
            f"{building_name_clean}*.webp",
            f"{building_name_clean}*.gif",
            f"{building_id}_*.jpg",
            f"{building_id}_*.jpeg",
            f"{building_id}_*.png",
            f"{building_id}_*.webp",
            f"{building_id}_*.gif",
            f"building_{building_id}*.jpg",
            f"building_{building_id}*.jpeg",
            f"building_{building_id}*.png",
            f"building_{building_id}*.webp",
            f"building_{building_id}*.gif"
        ]
        
        for pattern in patterns:
            for filepath in glob.glob(os.path.join(PHOTOS_DIR, pattern)):
                filename = os.path.basename(filepath)
                
                # Skip if already in list
                if filename not in [p['filename'] for p in photos]:
                    photos.append({
                        'filename': filename,
                        'url': f'/static/photos/{filename}',
                        'title': building_name,
                        'path': filepath,
                        'size': os.path.getsize(filepath),
                        'upload_time': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                    })
        
        # Update the building_photos dictionary
        building_photos[building_id] = photos
    
    print(f"✅ Refreshed photos for {len(building_photos)} buildings")
    
    # Count total photos
    total_photos = sum(len(photos) for photos in building_photos.values())
    print(f"📸 Total photos: {total_photos}")
    
    return building_photos

# ---------------------------------------------------
# REAL-TIME TRACKING FUNCTIONS
# ---------------------------------------------------
def cleanup_inactive_trackers():
    """Remove inactive trackers that haven't updated in TIMEOUT seconds"""
    current_time = time.time()
    inactive_trackers = []
    
    for tracker_id, tracker_data in list(active_trackers.items()):
        if current_time - tracker_data['timestamp'] > TRACKER_TIMEOUT:
            inactive_trackers.append(tracker_id)
    
    for tracker_id in inactive_trackers:
        if tracker_id in active_trackers:
            del active_trackers[tracker_id]
        if tracker_id in tracker_connections:
            del tracker_connections[tracker_id]
        print(f"🧹 Cleaned up inactive tracker: {tracker_id}")

def generate_tracker_id(user_id, tracker_name):
    """Generate a unique tracker ID"""
    timestamp = str(int(time.time()))
    unique_string = f"{user_id}_{tracker_name}_{timestamp}"
    return hashlib.md5(unique_string.encode()).hexdigest()[:8]

def get_tracker_color(tracker_id):
    """Get a consistent color for a tracker based on its ID"""
    hash_val = int(hashlib.md5(tracker_id.encode()).hexdigest()[:8], 16)
    return TRACKER_COLORS[hash_val % len(TRACKER_COLORS)]

def get_tracker_icon(tracker_id):
    """Get a consistent icon for a tracker based on its ID"""
    hash_val = int(hashlib.md5(tracker_id.encode()).hexdigest()[:8], 16)
    return TRACKER_ICONS[hash_val % len(TRACKER_ICONS)]

# ---------------------------------------------------
# SOCKET.IO EVENT HANDLERS
# ---------------------------------------------------
@socketio.on('connect')
def handle_connect():
    print(f"🔌 New client connected: {request.sid}")
    user_sessions[request.sid] = {'user_id': None, 'tracker_id': None}
    emit('connected', {'message': 'Connected to tracking server', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 Client disconnected: {request.sid}")
    
    if request.sid in user_sessions:
        session_data = user_sessions[request.sid]
        tracker_id = session_data.get('tracker_id')
        user_id = session_data.get('user_id')
        
        if tracker_id and tracker_id in tracker_connections:
            # Remove this connection from the tracker
            if request.sid in tracker_connections[tracker_id]:
                tracker_connections[tracker_id].remove(request.sid)
                
            # If no more connections for this tracker, mark user as offline
            if len(tracker_connections[tracker_id]) == 0 and tracker_id in active_trackers:
                active_trackers[tracker_id]['online'] = False
                
                # Broadcast user offline status
                emit('tracker_status', {
                    'tracker_id': tracker_id,
                    'online': False,
                    'timestamp': time.time()
                }, broadcast=True, room=tracker_id)
        
        del user_sessions[request.sid]

@socketio.on('create_tracker')
def handle_create_tracker(data):
    """Create a new location tracker"""
    try:
        user_id = data.get('user_id', 'anonymous')
        tracker_name = data.get('name', 'My Location')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        # Generate tracker ID
        tracker_id = generate_tracker_id(user_id, tracker_name)
        
        # Create tracker data
        tracker_data = {
            'user_id': user_id,
            'name': tracker_name,
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': time.time(),
            'color': get_tracker_color(tracker_id),
            'icon': get_tracker_icon(tracker_id),
            'mode': data.get('mode', 'walking'),
            'speed': data.get('speed', 0),
            'accuracy': data.get('accuracy', 0),
            'online': True
        }
        
        # Store tracker
        active_trackers[tracker_id] = tracker_data
        
        # Initialize connections list
        tracker_connections[tracker_id] = [request.sid]
        
        # Update user session
        user_sessions[request.sid] = {
            'user_id': user_id,
            'tracker_id': tracker_id
        }
        
        # Join tracker room
        join_room(tracker_id)
        
        # Send confirmation to creator
        emit('tracker_created', {
            'tracker_id': tracker_id,
            'tracker_data': tracker_data,
            'message': 'Tracker created successfully'
        })
        
        # Broadcast new tracker to all connected clients
        emit('tracker_update', {
            'tracker_id': tracker_id,
            'tracker_data': tracker_data,
            'action': 'create'
        }, broadcast=True)
        
        print(f"📍 New tracker created: {tracker_id} for user {user_id}")
        
    except Exception as e:
        print(f"Error creating tracker: {e}")
        emit('error', {'message': f'Failed to create tracker: {str(e)}'})

@socketio.on('join_tracker')
def handle_join_tracker(data):
    """Join an existing tracker to receive updates"""
    try:
        tracker_id = data.get('tracker_id')
        
        if tracker_id not in active_trackers:
            emit('error', {'message': 'Tracker not found or inactive'})
            return
        
        # Join the tracker room
        join_room(tracker_id)
        
        # Add to connections
        if request.sid not in tracker_connections[tracker_id]:
            tracker_connections[tracker_id].append(request.sid)
        
        # Send current tracker data to the joiner
        emit('tracker_joined', {
            'tracker_id': tracker_id,
            'tracker_data': active_trackers[tracker_id],
            'message': 'Joined tracker successfully'
        })
        
        # Update user session
        user_sessions[request.sid] = {
            'user_id': data.get('user_id', 'anonymous'),
            'tracker_id': tracker_id
        }
        
        print(f"👤 User joined tracker: {tracker_id}")
        
    except Exception as e:
        print(f"Error joining tracker: {e}")
        emit('error', {'message': f'Failed to join tracker: {str(e)}'})

@socketio.on('update_location')
def handle_update_location(data):
    """Update location for a tracker"""
    try:
        tracker_id = data.get('tracker_id')
        
        if tracker_id not in active_trackers:
            emit('error', {'message': 'Tracker not found'})
            return
        
        # Update tracker data
        active_trackers[tracker_id].update({
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': time.time(),
            'speed': data.get('speed', 0),
            'accuracy': data.get('accuracy', 0),
            'heading': data.get('heading'),
            'altitude': data.get('altitude'),
            'mode': data.get('mode', 'walking'),
            'online': True
        })
        
        # Broadcast update to all users in the tracker room
        emit('tracker_update', {
            'tracker_id': tracker_id,
            'tracker_data': active_trackers[tracker_id],
            'action': 'update'
        }, room=tracker_id)
        
        # Also broadcast to all connected clients for tracking display
        emit('tracker_location_update', {
            'tracker_id': tracker_id,
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': time.time(),
            'speed': data.get('speed', 0),
            'heading': data.get('heading')
        }, broadcast=True)
        
        # Log update
        # print(f"📍 Tracker {tracker_id} location updated: {data.get('latitude')}, {data.get('longitude')}")
        
    except Exception as e:
        print(f"Error updating location: {e}")
        emit('error', {'message': f'Failed to update location: {str(e)}'})

@socketio.on('leave_tracker')
def handle_leave_tracker(data):
    """Leave a tracker"""
    try:
        tracker_id = data.get('tracker_id')
        
        if tracker_id in tracker_connections and request.sid in tracker_connections[tracker_id]:
            tracker_connections[tracker_id].remove(request.sid)
            leave_room(tracker_id)
            
            # If no more connections, mark tracker as offline
            if len(tracker_connections[tracker_id]) == 0:
                if tracker_id in active_trackers:
                    active_trackers[tracker_id]['online'] = False
                
                # Broadcast offline status
                emit('tracker_status', {
                    'tracker_id': tracker_id,
                    'online': False,
                    'timestamp': time.time()
                }, broadcast=True)
        
        # Clear user session
        if request.sid in user_sessions:
            del user_sessions[request.sid]
        
        emit('tracker_left', {'message': 'Left tracker successfully'})
        
    except Exception as e:
        print(f"Error leaving tracker: {e}")
        emit('error', {'message': f'Failed to leave tracker: {str(e)}'})

@socketio.on('get_active_trackers')
def handle_get_active_trackers():
    """Get all active trackers"""
    try:
        # Clean up inactive trackers first
        cleanup_inactive_trackers()
        
        emit('active_trackers', {
            'trackers': active_trackers,
            'count': len(active_trackers)
        })
        
    except Exception as e:
        print(f"Error getting active trackers: {e}")
        emit('error', {'message': f'Failed to get active trackers: {str(e)}'})

# ---------------------------------------------------
# API ENDPOINTS WITH PHOTO SUPPORT
# ---------------------------------------------------
route_history = []

@app.route('/api/route_history', methods=['GET', 'POST'])
def handle_route_history():
    if request.method == 'POST':
        try:
            data = request.json
            route_history.append({
                'timestamp': datetime.now().isoformat(),
                'start': data.get('start'),
                'end': data.get('end'),
                'mode': data.get('mode'),
                'distance': data.get('distance'),
                'time': data.get('time')
            })
            if len(route_history) > 20:
                route_history.pop(0)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({
            'history': route_history[-10:],
            'success': True
        })

@app.route('/api/building/<int:building_id>', methods=['GET'])
def get_building_details(building_id):
    try:
        building = df[df['id'] == building_id].iloc[0]
        
        # Get building photos (from your existing building_photos dictionary)
        photos = building_photos.get(building_id, [])
        
        # Ensure photos are properly formatted
        formatted_photos = []
        for photo in photos:
            if isinstance(photo, dict):
                formatted_photos.append({
                    'filename': photo.get('filename', ''),
                    'url': photo.get('url', f'/static/photos/{photo.get("filename", "")}'),
                    'title': photo.get('title', building['name']),
                    'uploaded_at': photo.get('uploaded_at', datetime.now().isoformat())
                })
        
        # Get nearby buildings (within 200m)
        nearby_buildings = []
        for _, other in df.iterrows():
            if other['id'] != building_id:
                distance = haversine_distance(
                    building['longitude'], building['latitude'],
                    other['longitude'], other['latitude']
                )
                if distance <= 200:
                    config = ICON_CONFIG.get(str(other['type']), ICON_CONFIG['default'])
                    nearby_buildings.append({
                        'id': int(other['id']),
                        'name': str(other['name']),
                        'type': str(other['type']),
                        'distance': round(distance, 1),
                        'icon': config['icon'],
                        'color': config['color']
                    })
        
        # Sort by distance
        nearby_buildings = sorted(nearby_buildings, key=lambda x: x['distance'])[:5]
        
        config = ICON_CONFIG.get(str(building['type']), ICON_CONFIG['default'])
        
        return jsonify({
            'success': True,
            'building': {
                'id': int(building['id']),
                'name': str(building['name']),
                'type': str(building['type']),
                'type_name': config.get('name', str(building['type']).replace('_', ' ').title()),
                'category': str(building['category']),
                'latitude': float(building['latitude']),
                'longitude': float(building['longitude']),
                'description': str(building['description']),
                'icon': config['icon'],
                'color': config['color'],
                'photos': formatted_photos,
                'photo_count': len(formatted_photos)
            },
            'nearby': nearby_buildings
        })
    except Exception as e:
        print(f"Error getting building details: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/upload_photo/<int:building_id>', methods=['POST'])
def upload_building_photo(building_id):
    """POST endpoint - DISABLED to prevent photo uploads"""
    return jsonify({
        'success': False,
        'error': 'Photo upload is disabled. Photos are view-only.'
    })

@app.route('/api/delete_photo/<int:building_id>/<filename>', methods=['DELETE'])
def delete_building_photo(building_id, filename):
    """DELETE endpoint - DISABLED to prevent photo deletion"""
    return jsonify({
        'success': False,
        'error': 'Photo deletion is disabled. Photos are view-only.'
    })

@app.route('/api/photos/<int:building_id>', methods=['GET'])
def get_building_photos_api(building_id):
    try:
        photos = building_photos.get(building_id, [])
        return jsonify({
            'success': True,
            'photos': photos,
            'count': len(photos)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/static/photos/<filename>')
def serve_photo(filename):
    return send_from_directory(PHOTOS_DIR, filename)

# ---------------------------------------------------
# GET ALL PATHS FOR MAP DISPLAY
# ---------------------------------------------------
@app.route('/api/paths', methods=['GET'])
def get_all_paths():
    """Get all paths for map display"""
    try:
        mode = request.args.get('mode', 'all')
        paths_data = get_paths(mode if mode != 'all' else None)
        
        geojson_features = []
        path_type_counts = {}
        
        for path_id, geojson_str, path_type, length, name in paths_data:
            try:
                if isinstance(geojson_str, str):
                    geojson = json.loads(geojson_str)
                else:
                    geojson = geojson_str
                
                feature = {
                    "type": "Feature",
                    "geometry": geojson,
                    "properties": {
                        "id": path_id,
                        "type": path_type,
                        "name": name or f"Path {path_id}",
                        "length": float(length) if length else 0,
                        "compatible_modes": get_compatible_modes(path_type)
                    }
                }
                geojson_features.append(feature)
                
                if path_type in path_type_counts:
                    path_type_counts[path_type] += 1
                else:
                    path_type_counts[path_type] = 1
                    
            except Exception as e:
                print(f"Error processing path {path_id}: {e}")
                continue
        
        print(f"✓ Processed {len(geojson_features)} path features for mode: {mode}")
        
        return jsonify({
            "type": "FeatureCollection",
            "features": geojson_features,
            "success": True,
            "count": len(geojson_features),
            "mode": mode,
            "type_counts": path_type_counts
        })
        
    except Exception as e:
        print(f"Error getting paths: {e}")
        return jsonify({
            "type": "FeatureCollection",
            "features": [],
            "success": False,
            "error": str(e)
        })

@app.route('/api/paths/<mode>', methods=['GET'])
def get_paths_by_mode(mode):
    """Get paths for a specific mode"""
    return get_all_paths()

def get_compatible_modes(path_type):
    """Get list of modes compatible with this path type"""
    compatible = []
    for mode, types in PATH_COMPATIBILITY.items():
        if path_type in types:
            compatible.append(mode)
    return compatible

# ---------------------------------------------------
# REAL-TIME TRACKING API ENDPOINTS
# ---------------------------------------------------

@app.route('/api/tracking/create', methods=['POST'])
def api_create_tracker():
    """API endpoint to create a tracker"""
    try:
        data = request.json
        user_id = data.get('user_id', 'anonymous')
        tracker_name = data.get('name', 'My Location')
        latitude = data.get('latitude')
        longitude = data.get('longitude')
        
        if not latitude or not longitude:
            return jsonify({'success': False, 'error': 'Latitude and longitude required'})
        
        # Generate tracker ID
        tracker_id = generate_tracker_id(user_id, tracker_name)
        
        # Create tracker data
        tracker_data = {
            'user_id': user_id,
            'name': tracker_name,
            'latitude': latitude,
            'longitude': longitude,
            'timestamp': time.time(),
            'color': get_tracker_color(tracker_id),
            'icon': get_tracker_icon(tracker_id),
            'mode': data.get('mode', 'walking'),
            'speed': data.get('speed', 0),
            'accuracy': data.get('accuracy', 0),
            'online': True
        }
        
        # Store tracker
        active_trackers[tracker_id] = tracker_data
        tracker_connections[tracker_id] = []
        
        return jsonify({
            'success': True,
            'tracker_id': tracker_id,
            'tracker_data': tracker_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/tracking/update/<tracker_id>', methods=['POST'])
def api_update_tracker(tracker_id):
    """API endpoint to update tracker location"""
    try:
        if tracker_id not in active_trackers:
            return jsonify({'success': False, 'error': 'Tracker not found'})
        
        data = request.json
        
        # Update tracker data
        active_trackers[tracker_id].update({
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': time.time(),
            'speed': data.get('speed', 0),
            'accuracy': data.get('accuracy', 0),
            'heading': data.get('heading'),
            'altitude': data.get('altitude'),
            'mode': data.get('mode', 'walking'),
            'online': True
        })
        
        # Emit socket update
        socketio.emit('tracker_update', {
            'tracker_id': tracker_id,
            'tracker_data': active_trackers[tracker_id],
            'action': 'update'
        }, room=tracker_id)
        
        socketio.emit('tracker_location_update', {
            'tracker_id': tracker_id,
            'latitude': data.get('latitude'),
            'longitude': data.get('longitude'),
            'timestamp': time.time(),
            'speed': data.get('speed', 0),
            'heading': data.get('heading')
        }, broadcast=True)
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/tracking/trackers', methods=['GET'])
def api_get_trackers():
    """API endpoint to get all active trackers"""
    cleanup_inactive_trackers()
    
    return jsonify({
        'success': True,
        'trackers': active_trackers,
        'count': len(active_trackers)
    })

@app.route('/api/tracking/tracker/<tracker_id>', methods=['GET'])
def api_get_tracker(tracker_id):
    """API endpoint to get specific tracker"""
    if tracker_id not in active_trackers:
        return jsonify({'success': False, 'error': 'Tracker not found'})
    
    return jsonify({
        'success': True,
        'tracker': active_trackers[tracker_id]
    })

@app.route('/api/tracking/delete/<tracker_id>', methods=['DELETE'])
def api_delete_tracker(tracker_id):
    """API endpoint to delete a tracker"""
    try:
        if tracker_id in active_trackers:
            # Broadcast deletion
            socketio.emit('tracker_update', {
                'tracker_id': tracker_id,
                'action': 'delete'
            }, broadcast=True)
            
            # Remove tracker
            del active_trackers[tracker_id]
            if tracker_id in tracker_connections:
                del tracker_connections[tracker_id]
            
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Tracker not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ---------------------------------------------------
# GET BUILDING DATA FOR LABELS
# ---------------------------------------------------
@app.route('/api/building_labels', methods=['GET'])
def get_building_labels():
    """API endpoint to get building data for map labels"""
    try:
        building_labels = []
        for _, row in df.iterrows():
            config = ICON_CONFIG.get(str(row['type']), ICON_CONFIG['default'])
            building_labels.append({
                'id': int(row['id']),
                'name': str(row['name']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'type': str(row['type']),
                'color': config['color'],
                'icon': config['icon'],
                'category': str(row['category'])
            })
        
        return jsonify({
            'success': True,
            'buildings': building_labels,
            'count': len(building_labels)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ---------------------------------------------------
# PATH ADMINISTRATION ENDPOINTS
# ---------------------------------------------------
@app.route('/api/debug/routes', methods=['GET'])
def debug_routes():
    """Debug endpoint to check route availability"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({'error': 'Database connection failed'})
        
        cursor = conn.cursor()
        
        # Check if embu_paths table exists and has data
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'embu_paths'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            return jsonify({'error': 'embu_paths table does not exist'})
        
        # Get path count by type
        cursor.execute("""
            SELECT path_type, COUNT(*) 
            FROM embu_paths 
            GROUP BY path_type
        """)
        path_counts = cursor.fetchall()
        
        # Get sample path
        cursor.execute("""
            SELECT id, name, path_type, 
                   ST_AsGeoJSON(geom) as geojson
            FROM embu_paths 
            LIMIT 1
        """)
        sample = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'table_exists': table_exists,
            'path_counts': [{'type': row[0], 'count': row[1]} for row in path_counts],
            'sample_path': sample[3] if sample else None,
            'total_paths': sum(row[1] for row in path_counts)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/admin/edit_paths', methods=['GET'])
@token_required
def edit_paths_page(current_user):
    """Page for editing paths"""
    return render_template('edit_paths.html', user=current_user)

@app.route('/api/get_all_paths', methods=['GET'])
def get_all_paths_for_editing():
    """Get all paths with details for editing"""
    try:
        conn = get_db_connection()
        if conn is None:
            return jsonify({
                'success': False, 
                'error': 'Database connection failed',
                'paths': []
            })
        
        cursor = conn.cursor()
        
        # First check if the table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'embu_paths'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            cursor.close()
            conn.close()
            return jsonify({
                'success': False, 
                'error': 'Table embu_paths does not exist',
                'paths': []
            })
        
        # Get all paths
        cursor.execute("""
            SELECT 
                id,
                COALESCE(name, '') as name,
                COALESCE(path_type, 'walking') as path_type,
                ST_AsGeoJSON(geom) as geojson,
                ROUND(ST_Length(geom::geography)::numeric, 2) as length,
                ST_X(ST_StartPoint(geom)) as start_lon,
                ST_Y(ST_StartPoint(geom)) as start_lat,
                ST_X(ST_EndPoint(geom)) as end_lon,
                ST_Y(ST_EndPoint(geom)) as end_lat
            FROM embu_paths 
            ORDER BY id
        """)
        
        paths = []
        for row in cursor.fetchall():
            try:
                path = {
                    'id': row[0],
                    'name': row[1] or f"Path {row[0]}",
                    'type': row[2] or 'walking',
                    'geojson': row[3],
                    'length': row[4],
                    'start': [row[6], row[5]] if row[5] and row[6] else None,  # lat, lon
                    'end': [row[8], row[7]] if row[7] and row[8] else None     # lat, lon
                }
                paths.append(path)
            except Exception as e:
                print(f"Error processing path {row[0]}: {e}")
                continue
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'paths': paths,
            'count': len(paths),
            'table_exists': table_exists
        })
        
    except Exception as e:
        print(f"Error in get_all_paths: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'paths': []
        })

@app.route('/api/update_path', methods=['POST'])
def update_path():
    """Update a path's type and name"""
    try:
        data = request.json
        path_id = data.get('id')
        path_type = data.get('type')
        name = data.get('name')
        
        if not path_id:
            return jsonify({'success': False, 'error': 'Missing path ID'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embu_paths 
            SET path_type = %s, 
                name = %s
            WHERE id = %s
        """, (path_type, name, path_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/paths', methods=['GET'])
@token_required
def admin_paths(current_user):
    """Admin interface to view and update paths"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, path_type, 
                   ST_Length(geom::geography) as length,
                   ST_AsText(ST_StartPoint(geom)) as start_point,
                   ST_AsText(ST_EndPoint(geom)) as end_point
            FROM embu_paths
            ORDER BY id
        """)
        
        paths = []
        for row in cursor.fetchall():
            paths.append({
                'id': row[0],
                'name': row[1] or f"Path {row[0]}",
                'type': row[2] or 'unknown',
                'length': round(float(row[3]), 2) if row[3] else 0,
                'start_point': row[4],
                'end_point': row[5]
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'paths': paths,
            'count': len(paths)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_path_type', methods=['POST'])
def update_path_type():
    """Update path type from admin interface"""
    try:
        data = request.json
        path_id = data.get('path_id')
        path_type = data.get('path_type')
        
        if not path_id or not path_type:
            return jsonify({'success': False, 'error': 'Missing parameters'})
        
        if path_type not in ['walking', 'cycling', 'driving']:
            return jsonify({'success': False, 'error': 'Invalid path type'})
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embu_paths 
            SET path_type = %s 
            WHERE id = %s
        """, (path_type, path_id))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ---------------------------------------------------
# DEBUG ENDPOINTS
# ---------------------------------------------------

@app.route('/api/debug/paths', methods=['GET'])
def debug_paths():
    """Debug endpoint to check path data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'embu_paths'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            return jsonify({'error': 'Table embu_paths does not exist'})
        
        # Get column names
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'embu_paths' 
            ORDER BY ordinal_position;
        """)
        columns = cursor.fetchall()
        
        # Count paths
        cursor.execute("SELECT COUNT(*) FROM embu_paths;")
        count = cursor.fetchone()[0]
        
        # Get sample data
        cursor.execute("""
            SELECT id, name, path_type, 
                   ST_AsGeoJSON(geom) as geojson
            FROM embu_paths 
            LIMIT 5;
        """)
        sample_data = []
        for row in cursor.fetchall():
            sample_data.append({
                'id': row[0],
                'name': row[1],
                'path_type': row[2],
                'has_geojson': bool(row[3])
            })
        
        cursor.close()
        conn.close()
        
        return jsonify({
            'table_exists': table_exists,
            'column_count': len(columns),
            'columns': [{'name': c[0], 'type': c[1]} for c in columns],
            'total_paths': count,
            'sample_data': sample_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# ---------------------------------------------------
# BULK PHOTO UPLOAD
# ---------------------------------------------------

@app.route('/admin/bulk_photos', methods=['GET'])
@token_required
def bulk_photos_page(current_user):
    """Page for bulk photo upload"""
    # Make sure you're passing the buildings data
    building_data = []
    for _, row in df.iterrows():
        cfg = ICON_CONFIG.get(str(row['type']), ICON_CONFIG['default'])
        building_data.append({
            'id': int(row['id']),
            'name': str(row['name']),
            'type_name': cfg.get('name', str(row['type']).replace('_', ' ').title()),
            'description': str(row['description']),
            'color': cfg['color'],
            'icon': cfg['icon']
        })
    
    return render_template('bulk_photos.html', buildings=building_data, user=current_user)

@app.route('/api/bulk_upload_photos', methods=['POST'])
def bulk_upload_photos():
    """Bulk upload photos for multiple buildings - FIXED VERSION"""
    try:
        building_id = request.form.get('building_id')
        
        if not building_id:
            return jsonify({'success': False, 'error': 'No building ID provided'})
        
        photos = request.files.getlist('photos')
        
        if not photos or len(photos) == 0:
            return jsonify({'success': False, 'error': 'No photos uploaded'})
        
        # Check if building exists
        if int(building_id) not in df['id'].values:
            return jsonify({'success': False, 'error': f'Building ID {building_id} does not exist'})
        
        building = df[df['id'] == int(building_id)].iloc[0]
        building_name = str(building['name'])
        building_name_clean = building_name.lower().replace(' ', '_').replace('.', '').replace(',', '')
        
        uploaded_files = []
        
        for photo in photos:
            if photo and allowed_file(photo.filename):
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                file_ext = photo.filename.rsplit('.', 1)[1].lower() if '.' in photo.filename else 'jpg'
                
                # Create filename pattern: buildingid_buildingname_timestamp.ext
                filename = f"{building_id}_{building_name_clean}_{timestamp}.{file_ext}"
                filename = secure_filename(filename)
                
                filepath = os.path.join(PHOTOS_DIR, filename)
                
                # Save the file
                try:
                    photo.save(filepath)
                    
                    # Verify the file was saved
                    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                        
                        # Add to uploaded files list
                        uploaded_files.append(filename)
                        
                        # Add to building_photos dictionary immediately
                        if int(building_id) not in building_photos:
                            building_photos[int(building_id)] = []
                        
                        # Check if photo already exists
                        if not any(p['filename'] == filename for p in building_photos[int(building_id)]):
                            building_photos[int(building_id)].append({
                                'filename': filename,
                                'url': f'/static/photos/{filename}',
                                'title': building_name,
                                'uploaded_at': datetime.now().isoformat(),
                                'size': os.path.getsize(filepath)
                            })
                            
                            print(f"✅ Added photo: {filename} to building {building_id}")
                        else:
                            print(f"⚠ Photo already exists: {filename}")
                    else:
                        print(f"⚠ Failed to save photo: {filename}")
                        
                except Exception as e:
                    print(f"⚠ Error saving photo {photo.filename}: {e}")
                    continue
        
        if uploaded_files:
            # Force refresh of building photos
            refresh_building_photos()
            
            # Save to a log file for debugging
            log_file = os.path.join(BASE_DIR, 'photo_uploads.log')
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - Building {building_id}: Uploaded {len(uploaded_files)} photos: {uploaded_files}\n")
            
            return jsonify({
                'success': True,
                'uploaded_count': len(uploaded_files),
                'files': uploaded_files,
                'building_id': building_id,
                'building_name': building_name,
                'total_photos': len(building_photos.get(int(building_id), []))
            })
        else:
            return jsonify({'success': False, 'error': 'No valid photos were uploaded'})
        
    except Exception as e:
        print(f"✗ Bulk upload error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

        
# ---------------------------------------------------
# TEST ROUTE FOR MAP DEBUGGING
# ---------------------------------------------------
@app.route('/test')
def test_map():
    """Test route to check if Leaflet loads"""
    return render_template('test_map.html')

@app.route('/api/ai/query', methods=['POST'])
def ai_query():
    """AI-powered query endpoint for natural language navigation"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'No query provided'
            })
        
        print(f"🤖 Processing AI query: {query}")
        
        # Check if NLP processor is available
        if nlp_processor is None:
            # Fallback response
            return jsonify({
                'success': False,
                'error': 'AI processor not available',
                'response': "I'm currently learning about the campus. Please try again in a moment."
            })
        
        # Process query with NLP
        result = nlp_processor.process_query(query)
        
        # If building found, get its details
        building_details = None
        if result.get('building'):
            building = df[df['name'].str.lower() == result['building'].lower()]
            if not building.empty:
                building = building.iloc[0]
                config = ICON_CONFIG.get(str(building['type']), ICON_CONFIG['default'])
                building_details = {
                    'id': int(building['id']),
                    'name': str(building['name']),
                    'latitude': float(building['latitude']),
                    'longitude': float(building['longitude']),
                    'type': str(building['type']),
                    'type_name': config.get('name', str(building['type']).replace('_', ' ').title())
                }
        
        response_data = {
            'success': result.get('success', False),
            'original_query': result.get('original_query', query),
            'intent': result.get('intent', 'unknown'),
            'intent_confidence': result.get('intent_confidence', 0.0),
            'building': result.get('building'),
            'building_confidence': result.get('building_confidence', 0.0),
            'response': result.get('response', "I'm not sure how to answer that."),
            'context': result.get('context', {}),
            'building_details': building_details,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing AI query: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I encountered an error processing your request. Please try again."
        })

# Add voice input endpoint
@app.route('/api/ai/voice', methods=['POST'])
def voice_input():
    """Handle voice input (simplified - would need actual speech recognition)"""
    try:
        # In a real implementation, this would process audio files
        # For now, we'll accept transcribed text
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No voice text provided'
            })
        
        # Process as regular query
        return ai_query()
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add endpoint to test NLP
@app.route('/api/ai/test', methods=['GET'])
def test_nlp():
    """Test endpoint for NLP functionality"""
    if nlp_processor is None:
        return jsonify({
            'success': False,
            'error': 'NLP processor not initialized'
        })
    
    test_queries = [
        "Where is the library?",
        "How do I get to administration building?",
        "Hello, can you help me?",
        "Show me the route to the cafeteria",
        "What is the computer lab?",
        "Thank you for your help"
    ]
    
    results = []
    for query in test_queries:
        result = nlp_processor.process_query(query)
        results.append(result)
    
    return jsonify({
        'success': True,
        'test_results': results,
        'total_queries': len(results),
        'processor_type': nlp_processor.__class__.__name__
    })

@app.route('/ai-chat')
@token_required
def ai_chat(current_user):
    """Serve the AI chat interface"""
    return render_template('chat_interface.html', user=current_user)

@app.route('/api/ai/advanced_query', methods=['POST'])
def advanced_ai_query():
    """Advanced AI-powered query with voice and multilingual support"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        is_voice = data.get('is_voice', False)
        language = data.get('language', 'auto')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'No query provided'
            })
        
        print(f"🤖 Processing advanced AI query: {query}")
        print(f"   Voice: {is_voice}, Language: {language}")
        
        # Process with advanced NLP
        result = nlp_processor.process_query(query, is_voice=is_voice)
        
        # Add route information if building found
        if result.get('building') and result.get('building_details'):
            building = result['building_details']
            
            # Get current location (if provided)
            current_lat = data.get('current_lat')
            current_lon = data.get('current_lon')
            
            if current_lat and current_lon:
                # Calculate route
                try:
                    response = requests.get(
                        f"http://localhost:5000/route?start_lat={current_lat}&start_lon={current_lon}&end_lat={building['latitude']}&end_lon={building['longitude']}&mode={result['context'].get('mode', 'walking')}"
                    )
                    if response.status_code == 200:
                        route_data = response.json()
                        result['route'] = route_data
                except:
                    pass
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error processing advanced AI query: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "I encountered an error processing your advanced request."
        })

@app.route('/api/ai/voice/process', methods=['POST'])
def process_voice_input():
    """Process voice input from audio file"""
    try:
        # Check if audio file is provided
        if 'audio' in request.files:
            audio_file = request.files['audio']
            
            # Save temporary file
            temp_path = os.path.join('/tmp', f'voice_{int(time.time())}.wav')
            audio_file.save(temp_path)
            
            # Convert speech to text
            text = voice_to_text(temp_path, language=request.form.get('language', 'en-US'))
            
            # Clean up
            os.remove(temp_path)
            
            if text:
                # Process the text
                result = nlp_processor.process_query(text, is_voice=True)
                return jsonify(result)
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not recognize speech'
                })
        
        # If text is provided directly
        elif request.json and 'text' in request.json:
            text = request.json['text']
            result = nlp_processor.process_query(text, is_voice=True)
            return jsonify(result)
        
        else:
            return jsonify({
                'success': False,
                'error': 'No audio or text provided'
            })
            
    except Exception as e:
        print(f"Error processing voice input: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ai/languages', methods=['GET'])
def get_supported_languages():
    """Get supported languages"""
    return jsonify({
        'success': True,
        'languages': [
            {'code': 'en', 'name': 'English', 'native': 'English'},
            {'code': 'sw', 'name': 'Swahili', 'native': 'Kiswahili'}
        ]
    })

@app.route('/api/ai/voice/commands', methods=['GET'])
def get_voice_commands():
    """Get supported voice commands"""
    try:
        processor = get_voice_processor()
        commands = processor.get_supported_commands()
        
        return jsonify({
            'success': True,
            'commands': commands,
            'examples': [
                "Where is the library?",
                "How do I get to administration?",
                "Iko wapi maktaba?",
                "Naenda wapi ofisi ya utawala?",
                "Take me to the cafeteria",
                "Find computer lab number 3"
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/ai/context/accessible', methods=['GET'])
def get_accessible_routes():
    """Get information about accessible routes"""
    return jsonify({
        'success': True,
        'accessible_features': [
            'Wheelchair-friendly paths',
            'Elevator access in buildings',
            'Ramp entrances',
            'Accessible restrooms',
            'Priority parking'
        ],
        'avoid_stairs': True,
        'recommended_mode': 'walking'
    })

@app.route('/api/ai/train/feedback', methods=['POST'])
def train_with_feedback():
    """Train AI model with user feedback"""
    try:
        data = request.json
        query = data.get('query')
        correct_intent = data.get('correct_intent')
        correct_building = data.get('correct_building')
        
        if not query or not correct_intent:
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            })
        
        # Train model with feedback
        nlp_processor.train_on_feedback(query, correct_intent, correct_building)
        
        return jsonify({
            'success': True,
            'message': 'Feedback received for model improvement'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Add TensorFlow model info endpoint
@app.route('/api/ai/models/info', methods=['GET'])
def get_model_info():
    """Get information about ML models"""
    try:
        return jsonify({
            'success': True,
            'models': {
                'intent_classification': {
                    'type': 'Neural Network',
                    'layers': 3,
                    'accuracy': '92% (estimated)',
                    'training_samples': 500
                },
                'building_recognition': {
                    'type': 'Neural Network',
                    'layers': 3,
                    'accuracy': '88% (estimated)',
                    'vocabulary_size': len(nlp_processor.campus_buildings)
                }
            },
            'languages_supported': ['English', 'Swahili'],
            'features': [
                'Natural Language Understanding',
                'Voice Command Processing',
                'Multilingual Support',
                'Context Awareness',
                'Accessibility Routing'
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# ---------------------------------------------------
# AUTHENTICATION ROUTES
# ---------------------------------------------------

@app.route('/login', methods=['GET'])
def login_page():
    """Serve login page"""
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login requests"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        user = USERS.get(username)
        
        if user and user['password'] == password:
            # Generate JWT token
            token = jwt.encode({
                'username': username,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, JWT_SECRET, algorithm='HS256')
            
            response = jsonify({
                'success': True, 
                'user': {
                    'username': username,
                    'name': user['name'],
                    'role': user['role']
                }
            })
            response.set_cookie('token', token, httponly=True, max_age=86400)
            return response
        
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle logout"""
    response = jsonify({'success': True})
    response.delete_cookie('token')
    return response

@app.route('/api/current_user', methods=['GET'])
def get_current_user():
    """Get current authenticated user"""
    token = request.cookies.get('token')
    
    if not token:
        return jsonify({'success': False, 'authenticated': False})
    
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = USERS.get(data['username'])
        if user:
            return jsonify({
                'success': True,
                'authenticated': True,
                'user': {
                    'username': data['username'],
                    'name': user['name'],
                    'role': user['role']
                }
            })
        else:
            return jsonify({'success': False, 'authenticated': False})
    except:
        return jsonify({'success': False, 'authenticated': False})

# ---------------------------------------------------
# MAIN ROUTE - PROTECTED
# ---------------------------------------------------
@app.route('/')
@token_required
def index(current_user):
    """Main map page with buildings and navigation"""
    
    # Get paths for display
    try:
        paths_data = get_paths()
        geojson_features = []
        path_type_counts = {}
        
        for path_id, geojson_str, path_type, length, name in paths_data:
            try:
                if isinstance(geojson_str, str):
                    geojson = json.loads(geojson_str)
                else:
                    geojson = geojson_str
                
                feature = {
                    "type": "Feature",
                    "geometry": geojson,
                    "properties": {
                        "id": path_id,
                        "type": path_type,
                        "name": name or f"Path {path_id}",
                        "length": float(length) if length else 0
                    }
                }
                geojson_features.append(feature)
                
                if path_type in path_type_counts:
                    path_type_counts[path_type] += 1
                else:
                    path_type_counts[path_type] = 1
                    
            except Exception as e:
                print(f"Error processing path {path_id}: {e}")
                continue
        
        print(f"✓ Loaded {len(geojson_features)} path segments for display")
        
    except Exception as e:
        print(f"✗ Error loading paths for display: {e}")
        geojson_features = []
        path_type_counts = {}

    # Convert building data
    building_data = []
    for _, row in df.iterrows():
        try:
            cfg = ICON_CONFIG.get(str(row['type']), ICON_CONFIG['default'])
            
            # Get photos for this building
            photos = building_photos.get(int(row['id']), [])
            has_photos = len(photos) > 0
            
            building = {
                'id': int(row['id']),
                'name': str(row['name']),
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'type': str(row['type']),
                'category': str(row['category']),
                'description': str(row['description']),
                'icon': cfg['icon'],
                'color': cfg['color'],
                'type_name': cfg.get('name', str(row['type']).replace('_', ' ').title()),
                'has_photos': has_photos,
                'photo_count': len(photos),
                'first_photo': photos[0]['url'] if photos else None
            }
            building_data.append(building)
        except Exception as e:
            print(f"Error processing building {row.get('name', 'Unknown')}: {e}")
            continue

    print(f"✓ Rendering template with {len(building_data)} buildings")
    
    # Get unique types for filter
    unique_types = sorted(set(df['type'].unique()))
    
    # Calculate campus center
    if len(building_data) > 0:
        center_lat = sum(b['latitude'] for b in building_data) / len(building_data)
        center_lon = sum(b['longitude'] for b in building_data) / len(building_data)
    else:
        center_lat, center_lon = -0.5075, 37.4575
    
    # Test pathfinder for each mode
    pathfinder_stats = {}
    for mode in ['walking', 'cycling', 'driving']:
        paths = get_paths(mode)
        pathfinder = SimplePathFinder(paths, mode)
        pathfinder_stats[mode] = {
            'segments': len(pathfinder.path_segments),
            'nodes': len(pathfinder.nodes),
            'graph_nodes': len(pathfinder.graph)
        }
    
    return render_template(
        'map.html',
        buildings=building_data,
        paths=geojson_features,
        ICON_CONFIG=ICON_CONFIG,
        SPEEDS=SPEEDS,
        ROUTE_COLORS=ROUTE_COLORS,
        MAP_TILES=MAP_TILES,
        UNIQUE_TYPES=unique_types,
        PATHFINDER_STATS=pathfinder_stats,
        TOTAL_BUILDINGS=len(building_data),
        TOTAL_PATHS=len(geojson_features),
        CAMPUS_CENTER=[center_lat, center_lon],
        PATH_TYPE_COUNTS=path_type_counts,
        PATH_COMPATIBILITY=PATH_COMPATIBILITY,
        current_user=current_user
    )

# ---------------------------------------------------
# CLEANUP THREAD FOR INACTIVE TRACKERS
# ---------------------------------------------------
def cleanup_worker():
    """Background thread to clean up inactive trackers"""
    while True:
        try:
            cleanup_inactive_trackers()
            time.sleep(60)  # Check every minute
        except Exception as e:
            print(f"Error in cleanup worker: {e}")
            time.sleep(60)

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
cleanup_thread.start()

## ---------------------------------------------------
# RUN SERVER WITH SOCKET.IO - CORRECTED FOR RENDER
# ---------------------------------------------------
if __name__ == '__main__':
    # This block runs ONLY for local development with 'python app.py'
    print("=" * 60)
    print("UNIVERSITY OF EMBU STARLIFE NAVIGATION SYSTEM (LOCAL DEVELOPMENT)")
    print("=" * 60)
    print(f"📊 Buildings: {len(df)}")
    print(f"🏛️  Building types: {len(df['type'].unique())}")
    print(f"📁 Categories: {len(df['category'].unique())}")
    print(f"📸 Total photos: {sum(len(photos) for photos in building_photos.values())}")
    print(f"🚶 Travel modes: {list(SPEEDS.keys())}")
    print(f"🛣️  Path types: walking, cycling, driving")
    print(f"🗺️  Map tiles: {len(MAP_TILES)} styles available")
    print(f"📍 Real-time tracking: ENABLED")
    print(f"📡 Socket.IO server: READY")
    print(f"🔐 Authentication: JWT-enabled")
    
    # Test database
    conn = get_db_connection()
    if conn:
        print("✓ Database connection successful")
        
        # Check path types distribution
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT path_type, COUNT(*) 
                FROM embu_paths 
                GROUP BY path_type 
                ORDER BY path_type
            """)
            print("📊 Path type distribution:")
            for path_type, count in cursor.fetchall():
                print(f"   {path_type}: {count} paths")
        except Exception as e:
            print(f"⚠ Error checking path types: {e}")
        finally:
            cursor.close()
            conn.close()
    else:
        print("⚠ Database connection failed - using sample data")
    
    # Test path loading for each mode
    print("\n🛣️  Testing path loading by mode...")
    for mode in ['walking', 'cycling', 'driving']:
        paths = get_paths(mode)
        print(f"   {mode}: {len(paths)} paths")
    
    # Get port from environment variable
    port = int(os.environ.get('PORT', 5000))
    
    print(f"\n🌐 Server starting on port {port}")
    print("📡 Socket.IO server running")
    print("=" * 60)
    
    # For local development - debug mode ON
    socketio.run(app, 
                debug=True,           # Enable debug mode for local development
                host='0.0.0.0', 
                port=port,
                allow_unsafe_werkzeug=True)  # Allow debug in development

# No else block needed - Gunicorn handles production on Render
# The SocketIO object is already configured with async_mode='eventlet'

# At the very end of app.py, after everything but before the if __name__ block
print("✅✅✅ DEBUG: App.py fully loaded and parsed", file=sys.stderr)