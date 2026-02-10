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

# ================= GOOGLE DRIVE FILES (UPDATED IDs) =================
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ✅ IDs CORRECTS
MODEL_ID = "1ZkmGBiDrgHO-UpCJ1r0jR1lR6Hzo4I1c"  # PKL file
DATA_ID  = "1e1oB9sGsdI7YtBUo_VnL4-S-fe5hEKJw"  # CSV file

MODEL_PATH = "models/model.pkl"
CSV_PATH   = "data/data.csv"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)
        print("✅ Model downloaded")
    except Exception as e:
        print(f"❌ Model download failed: {e}")

if not os.path.exists(CSV_PATH):
    print("⬇️ Downloading dataset...")
    try:
        gdown.download(f"https://drive.google.com/uc?id={DATA_ID}", CSV_PATH, quiet=False)
        print("✅ Dataset downloaded")
    except Exception as e:
        print(f"❌ Dataset download failed: {e}")

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
    return render_template(
        "index.html",
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

# ================= DASHBOARD =================
@app.route("/dashboard")
@login_required
def dashboard():
    txs = Transaction.query.filter_by(user_id=current_user.id).all()
    
    total_transactions = len(txs)
    total_frauds = sum(1 for t in txs if t.prediction == 1)
    total_normal = total_transactions - total_frauds
    fraud_percentage = (total_frauds / total_transactions * 100) if total_transactions > 0 else 0
    
    # Pie chart
    pie_html = ""
    if total_transactions > 0:
        fig = px.pie(
            values=[total_normal, total_frauds],
            names=["Normal", "Fraude"],
            color_discrete_sequence=["#34C759", "#FF3B30"]
        )
        pie_html = fig.to_html(full_html=False)
    
    # Gender chart
    gender_html = ""
    if txs:
        gender_data = pd.DataFrame([{"gender": t.gender, "prediction": t.prediction} for t in txs])
        fig = px.histogram(gender_data, x="gender", color="prediction", barmode="group")
        gender_html = fig.to_html(full_html=False)
    
    # Time series
    time_html = ""
    if txs:
        time_data = pd.DataFrame([{"date": t.created_at, "prediction": t.prediction} for t in txs])
        fig = px.line(time_data.groupby("date").sum().reset_index(), x="date", y="prediction")
        time_html = fig.to_html(full_html=False)
    
    return render_template(
        "dashboard.html",
        total_transactions=total_transactions,
        total_frauds=total_frauds,
        total_normal=total_normal,
        fraud_percentage=fraud_percentage,
        pie_html=pie_html,
        gender_html=gender_html,
        time_html=time_html
    )

# ================= ANALYSIS =================
@app.route("/analysis", methods=["GET", "POST"])
@login_required
def analysis():
    if not os.path.exists(CSV_PATH):
        flash("Data file not found", "danger")
        return redirect(url_for("home"))
    
    df = pd.read_csv(CSV_PATH)
    graph_html = None
    
    if request.method == "POST":
        analysis_type = request.form.get("analysis_type")
        col1 = request.form.get("col1")
        col2 = request.form.get("col2")
        graph_type = request.form.get("graph_type")
        
        if graph_type == "histogram" and col1:
            fig = px.histogram(df, x=col1)
            graph_html = fig.to_html(full_html=False)
        elif graph_type == "scatter" and col1 and col2:
            fig = px.scatter(df, x=col1, y=col2)
            graph_html = fig.to_html(full_html=False)
        elif graph_type == "box" and col1:
            fig = px.box(df, y=col1)
            graph_html = fig.to_html(full_html=False)
        elif graph_type == "bar" and col1:
            fig = px.bar(df, x=col1)
            graph_html = fig.to_html(full_html=False)
        elif graph_type == "pie" and col1:
            fig = px.pie(df, names=col1)
            graph_html = fig.to_html(full_html=False)
    
    return render_template(
        "analysis.html",
        df_cols=df.columns.tolist(),
        graph_html=graph_html,
        username=current_user.username
    )

# ================= USER ANALYSIS =================
@app.route("/user_analysis", methods=["GET", "POST"])
@login_required
def user_analysis():
    txs = Transaction.query.filter_by(user_id=current_user.id).all()
    user_df = pd.DataFrame([{
        "category": t.category,
        "amt": t.amt,
        "gender": t.gender,
        "city": t.city,
        "state": t.state,
        "job": t.job,
        "prediction": t.prediction
    } for t in txs])
    
    # Default graphs
    pie_html = ""
    box_html = ""
    custom_html = ""
    
    if not user_df.empty:
        # Pie
        pred_counts = user_df["prediction"].value_counts()
        fig = px.pie(values=pred_counts.values, names=pred_counts.index)
        pie_html = fig.to_html(full_html=False)
        
        # Box
        fig = px.box(user_df, y="amt")
        box_html = fig.to_html(full_html=False)
        
        # Custom
        if request.method == "POST":
            u_col1 = request.form.get("u_col1")
            u_col2 = request.form.get("u_col2")
            u_graph_type = request.form.get("u_graph_type")
            
            if u_graph_type == "histogram" and u_col1:
                fig = px.histogram(user_df, x=u_col1)
                custom_html = fig.to_html(full_html=False)
            elif u_graph_type == "scatter" and u_col1 and u_col2:
                fig = px.scatter(user_df, x=u_col1, y=u_col2)
                custom_html = fig.to_html(full_html=False)
            elif u_graph_type == "box" and u_col1:
                fig = px.box(user_df, y=u_col1)
                custom_html = fig.to_html(full_html=False)
    
    return render_template(
        "user_analysis.html",
        username=current_user.username,
        pie_html=pie_html,
        box_html=box_html,
        custom_html=custom_html,
        user_cols=user_df.columns.tolist() if not user_df.empty else []
    )

# ================= RUN =================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=3000, debug=True)