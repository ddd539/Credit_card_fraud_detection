import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pickle
import pandas as pd
import gdown

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, logout_user,
    login_required, current_user, UserMixin
)

import plotly.express as px
import plotly.figure_factory as ff

# ================= APP CONFIG =================
app = Flask(__name__)
app.config["SECRET_KEY"] = "change_this_secret_key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fraud.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# ================= GOOGLE DRIVE FILES =================
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

MODEL_ID = "1AmolxVsKZWQF-FWjWqxo_buzxrYCIvD9"
DATA_ID  = "1kHLflln8w0iOuBMr96cX1jQMqi1KNkU3"

MODEL_PATH = "models/model.pkl"
CSV_PATH   = "data/data.csv"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(CSV_PATH):
    print("⬇️ Downloading dataset...")
    gdown.download(f"https://drive.google.com/uc?id={DATA_ID}", CSV_PATH, quiet=False)

# ================= LOAD MODEL =================
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model loaded")
except Exception as e:
    print("❌ Model load error:", e)
    model = None

# ================= LOAD CSV VALUES =================
CITIES, STATES, JOBS = [], [], []

def load_unique_values():
    global CITIES, STATES, JOBS
    try:
        df = pd.read_csv(CSV_PATH)
        CITIES = sorted(df["city"].dropna().unique().tolist())
        STATES = sorted(df["state"].dropna().unique().tolist())
        JOBS   = sorted(df["job"].dropna().unique().tolist())
        print("✅ CSV values loaded")
    except Exception as e:
        print("❌ CSV error:", e)

load_unique_values()

# ================= DATABASE MODELS =================
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    category = db.Column(db.String(100))
    amt = db.Column(db.Float)
    gender = db.Column(db.String(20))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    job = db.Column(db.String(100))
    city_pop = db.Column(db.Integer)
    merch_lat = db.Column(db.Float)
    merch_long = db.Column(db.Float)
    prediction = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ================= AUTH ROUTES =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if User.query.filter_by(username=username).first():
            flash("Username already exists", "danger")
            return redirect(url_for("register"))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("home"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.check_password(request.form["password"]):
            login_user(user)
            return redirect(url_for("home"))
        flash("Invalid credentials", "danger")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ================= HOME =================
@app.route("/")
@login_required
def home():
    history = (
        Transaction.query
        .filter_by(user_id=current_user.id)
        .order_by(Transaction.created_at.desc())
        .limit(10)
        .all()
    )
    return render_template(
        "index.html",
        history=history,
        cities=CITIES,
        states=STATES,
        jobs=JOBS,
        username=current_user.username
    )

# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if model is None:
        flash("Model not loaded", "danger")
        return redirect(url_for("home"))

    features = pd.DataFrame([{
        "category": request.form["category"],
        "amt": float(request.form["amt"]),
        "gender": request.form["gender"],
        "city": request.form["city"],
        "state": request.form["state"],
        "city_pop": int(request.form["city_pop"]),
        "job": request.form["job"],
        "merch_lat": float(request.form["merch_lat"]),
        "merch_long": float(request.form["merch_long"])
    }])

    prediction = int(model.predict(features)[0])

    tx = Transaction(
        user_id=current_user.id,
        prediction=prediction,
        **features.iloc[0].to_dict()
    )
    db.session.add(tx)
    db.session.commit()

    flash("❌ Fraud detected" if prediction == 1 else "✅ Normal transaction", "info")
    return redirect(url_for("home"))

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=3000, debug=True)