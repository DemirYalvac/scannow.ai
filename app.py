import os
import json
import secrets
import requests
from datetime import datetime
from flask import Flask, render_template_string, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# --- Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # change for production or use env var
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///scannow.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

model = MobileNetV2(weights='imagenet')

# --- Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    scans = db.relationship('Scan', backref='user', lazy=True)

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(200), nullable=False)
    predictions = db.Column(db.Text, nullable=False)  # JSON string
    products = db.Column(db.Text, nullable=False)     # JSON string
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# --- Login Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Helpers ---
def recognize(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return [(label, float(prob)) for (_, label, prob) in decoded]

def search_amazon(keyword):
    # Placeholder simple scraping, needs more work for production
    url = f"https://www.amazon.com/s?k={keyword.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    # This will be basic because Amazon blocks scraping often
    # Just return dummy data here:
    return [{'title': f'Amazon {keyword} product', 'price': '$9.99', 'link': url}]

def search_ebay(keyword):
    url = f"https://www.ebay.com/sch/i.html?_nkw={keyword.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    # Dummy result for demo
    return [{'title': f'eBay {keyword} item', 'price': '$7.77', 'link': url}]

def search_walmart(keyword):
    url = f"https://www.walmart.com/search/?query={keyword.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)
    # Dummy result for demo
    return [{'title': f'Walmart {keyword} product', 'price': '$8.88', 'link': url}]

def combined_search(keyword):
    results = []
    results.extend(search_amazon(keyword))
    results.extend(search_ebay(keyword))
    results.extend(search_walmart(keyword))
    # Sort by price (dummy, just keep as is)
    return results[:6]  # top 6

# --- Routes & Views ---

@app.route('/')
@login_required
def index():
    return render_template_string(HOME_HTML, user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password')
    return render_template_string(LOGIN_HTML)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        if User.query.filter_by(username=username).first():
            flash('Username already taken')
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        user = User(username=username, password_hash=hashed)
        db.session.add(user)
        db.session.commit()
        flash('Account created! Please log in.')
        return redirect(url_for('login'))
    return render_template_string(REGISTER_HTML)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    manual_search = request.form.get('manual_search', '').strip()
    file = request.files.get('file')
    predictions = []
    products = []
    filename = None

    if manual_search:
        products = combined_search(manual_search)
    elif file and file.filename != '':
        filename = secrets.token_hex(8) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        predictions = recognize(filepath)
        top_label = predictions[0][0]
        products = combined_search(top_label)

        # Save scan to DB
        scan = Scan(
            image_filename=filename,
            predictions=json.dumps(predictions),
            products=json.dumps(products),
            user_id=current_user.id
        )
        db.session.add(scan)
        db.session.commit()
    else:
        flash("Please upload an image or enter a search term.")
        return redirect(url_for('index'))

    return render_template_string(RESULTS_HTML, user=current_user, predictions=predictions, products=products, image_url=url_for('uploaded_file', filename=filename) if filename else None, manual_search=manual_search)

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/history')
@login_required
def history():
    scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    for scan in scans:
        scan.predictions = json.loads(scan.predictions)
        scan.products = json.loads(scan.products)
    return render_template_string(HISTORY_HTML, user=current_user, scans=scans)

# --- HTML Templates ---

HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>ScanNow - Home</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body { padding: 2rem; background: #f8f9fa; }
        #drop-area {
            border: 2px dashed #6c757d;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            color: #6c757d;
        }
        #drop-area.highlight {
            border-color: #198754;
            color: #198754;
        }
        #preview {
            max-height: 150px;
            margin-top: 1rem;
            display: none;
            border-radius: 10px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>Welcome, {{ user.username }}!</h1>
    <a href="{{ url_for('logout') }}" class="btn btn-danger float-end">Logout</a>
    <a href="{{ url_for('history') }}" class="btn btn-secondary float-end me-2">Scan History</a>
    <h3>Upload an image or search manually</h3>

    <form id="upload-form" method="POST" action="{{ url_for('upload') }}" enctype="multipart/form-data" class="mb-3">
        <div id="drop-area" onclick="document.getElementById('fileElem').click()">
            Drag & Drop Image Here or Click to Select
            <input type="file" id="fileElem" name="file" accept="image/*" style="display:none" onchange="showPreview(event)">
            <img id="preview" src="#" alt="Preview" />
        </div>
        <button type="submit" class="btn btn-success mt-3">Scan Image</button>
    </form>

    <form method="POST" action="{{ url_for('upload') }}">
        <div class="input-group mb-3">
            <input type="text" name="manual_search" class="form-control" placeholder="Or type product name to search manually" value="{{ manual_search|default('') }}">
            <button class="btn btn-primary" type="submit">Search</button>
        </div>
    </form>
</div>

<script>
    let dropArea = document.getElementById('drop-area');
    let fileElem = document.getElementById('fileElem');
    let preview = document.getElementById('preview');

    dropArea.addEventListener('dragenter', e => { e.preventDefault(); dropArea.classList.add('highlight'); });
    dropArea.addEventListener('dragover', e => { e.preventDefault(); dropArea.classList.add('highlight'); });
    dropArea.addEventListener('dragleave', e => { e.preventDefault(); dropArea.classList.remove('highlight'); });
    dropArea.addEventListener('drop', e => {
        e.preventDefault();
        dropArea.classList.remove('highlight');
        let files = e.dataTransfer.files;
        if(files.length > 0){
            fileElem.files = files;
            showPreview({target: {files: files}});
        }
    });

    function showPreview(event){
        let file = event.target.files[0];
        if(file){
            let reader = new FileReader();
            reader.onload = function(e){
                preview.src = e.target.result;
                preview.style.display = 'block';
            }
            reader.readAsDataURL(file);
        }
    }
</script>
</body>
</html>
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>ScanNow - Login</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body class="bg-light">
<div class="container" style="max-width: 400px; margin-top: 5rem;">
    <h2>Login to ScanNow</h2>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-danger">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="POST">
        <div class="mb-3">
            <label>Username</label>
            <input name="username" class="form-control" required />
        </div>
        <div class="mb-3">
            <label>Password</label>
            <input name="password" type="password" class="form-control" required />
        </div>
        <button class="btn btn-primary" type="submit">Login</button>
        <a href="{{ url_for('register') }}" class="btn btn-link">Register</a>
    </form>
</div>
</body>
</html>
"""

REGISTER_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>ScanNow - Register</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body class="bg-light">
<div class="container" style="max-width: 400px; margin-top: 5rem;">
    <h2>Create an Account</h2>
    {% with messages = get_flashed_messages() %}
      {% if messages %}
        <div class="alert alert-danger">{{ messages[0] }}</div>
      {% endif %}
    {% endwith %}
    <form method="POST">
        <div class="mb-3">
            <label>Username</label>
            <input name="username" class="form-control" required />
        </div>
        <div class="mb-3">
            <label>Password</label>
            <input name="password" type="password" class="form-control" required />
        </div>
        <button class="btn btn-success" type="submit">Register</button>
        <a href="{{ url_for('login') }}" class="btn btn-link">Login</a>
    </form>
</div>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>ScanNow - Results</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        .progress { height: 20px; }
    </style>
</head>
<body>
<div class="container mt-4">
    <a href="{{ url_for('index') }}" class="btn btn-secondary mb-3">← Back</a>
    <h2>Results for {% if manual_search %}"{{ manual_search }}"{% else %}your image{% endif %}</h2>

    {% if image_url %}
    <div>
        <img src="{{ image_url }}" alt="Uploaded image" style="max-height: 200px; border-radius: 10px;">
    </div>
    {% endif %}

    {% if predictions %}
    <h3>AI Predictions</h3>
    <ul class="list-group mb-4">
        {% for label, conf in predictions %}
        <li class="list-group-item">
            <b>{{ label }}</b> — {{ "%.2f"|format(conf * 100) }}%
            <div class="progress mt-1">
                <div class="progress-bar bg-success" role="progressbar" style="width: {{ conf * 100 }}%;" aria-valuenow="{{ conf * 100 }}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
        </li>
        {% endfor %}
    </ul>
    {% endif %}

    {% if products %}
    <h3>Product Matches</h3>
    {% for item in products %}
    <div class="card mb-3">
        <div class="card-body">
            <h5 class="card-title">{{ item.title }}</h5>
            <p class="card-text">Price: {{ item.price }}</p>
            <a href="{{ item.link }}" class="btn btn-primary" target="_blank">Buy Here</a>
        </div>
    </div>
    {% endfor %}
    {% else %}
    <p>No products found.</p>
    {% endif %}
</div>
</body>
</html>
"""

HISTORY_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>ScanNow - History</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
<div class="container mt-4">
    <a href="{{ url_for('index') }}" class="btn btn-secondary mb-3">← Back</a>
    <h2>Your Scan History</h2>
    {% if scans %}
    <div class="row">
        {% for scan in scans %}
        <div class="col-md-4">
            <div class="card mb-4">
                <img src="{{ url_for('uploaded_file', filename=scan.image_filename) }}" class="card-img-top" alt="Scan image" style="height:200px;object-fit:cover;">
                <div class="card-body">
                    <h5 class="card-title">Scanned at {{ scan.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</h5>
                    <p><b>Predictions:</b></p>
                    <ul>
                    {% for label, conf in scan.predictions %}
                        <li>{{ label }} — {{ '%.2f' % (conf * 100) }}%</li>
                    {% endfor %}
                    </ul>
                    <p><b>Products:</b></p>
                    <ul>
                    {% for p in scan.products %}
                        <li><a href="{{ p.link }}" target="_blank">{{ p.title }} - {{ p.price }}</a></li>
                    {% endfor %}
                    </ul>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    {% else %}
    <p>You have no scans yet.</p>
    {% endif %}
</div>
</body>
</html>
"""

# --- Run ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
    
