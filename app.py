import os
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import pickle
import pandas as pd

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin

import plotly.express as px
import plotly.figure_factory as ff

# ======== CONFIG ========
app = Flask(__name__)
app.config['SECRET_KEY'] = 'replace_this_with_random_secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:douaeimami@localhost/fraud_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ======== MODELS ========
class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(db.Model):
    __tablename__ = 'transactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
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
    user = db.relationship('User', backref=db.backref('transactions', lazy='dynamic'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ======== PATHS ========
MODEL_PATH = r"C:/Users/douae/OneDrive/Desktop/ADA/Dossier/rf_pipeline2.pkl"
CSV_PATH = r"C:/Users/douae/OneDrive/Desktop/ADA/Dossier/fraudTrain.csv"

# ======== CHARGER LE MODELE ML ========
try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Modèle chargé:", MODEL_PATH)
except Exception as e:
    print("❌ Impossible de charger le modèle:", e)
    model = None

# ======== CHARGER LES VALEURS UNIQUES UNE SEULE FOIS AU DEMARRAGE ========
CITIES = []
STATES = []
JOBS = []

def load_unique_values():
    """Charge les valeurs uniques du CSV une seule fois au démarrage"""
    global CITIES, STATES, JOBS
    try:
        if os.path.exists(CSV_PATH):
            print("📂 Chargement des valeurs uniques depuis le CSV...")
            df = pd.read_csv(CSV_PATH)
            
            # Extraire les valeurs uniques et les trier
            CITIES = sorted(df['city'].dropna().unique().tolist())
            STATES = sorted(df['state'].dropna().unique().tolist())
            JOBS = sorted(df['job'].dropna().unique().tolist())
            
            print(f"✅ Chargé: {len(CITIES)} villes, {len(STATES)} états, {len(JOBS)} métiers")
        else:
            print(f"❌ Fichier CSV introuvable: {CSV_PATH}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du CSV: {e}")

# Charger les valeurs au démarrage de l'application
load_unique_values()

# ======== ROUTES AUTH ========
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        if not username or not password:
            flash("Remplis tous les champs.", "warning")
            return render_template('register.html')
        if User.query.filter_by(username=username).first():
            flash("Nom d'utilisateur déjà pris.", "danger")
            return render_template('register.html')
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        login_user(user)
        flash("Inscription réussie. Bienvenue !", "success")
        return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash("Connexion réussie.", "success")
            return redirect(url_for('home'))
        flash("Nom d'utilisateur ou mot de passe invalide.", "danger")
        return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Déconnecté.", "info")
    return redirect(url_for('login'))

# ======== PAGES ========
@app.route('/')
@login_required
def home():
    last_tx = Transaction.query.filter_by(user_id=current_user.id)\
                               .order_by(Transaction.created_at.desc())\
                               .limit(12).all()
    
    return render_template('index.html', 
                         history=last_tx, 
                         prediction=None, 
                         username=current_user.username, 
                         cities=CITIES, 
                         states=STATES, 
                         jobs=JOBS)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        flash("Modèle non chargé.", "danger")
        return redirect(url_for('home'))
    try:
        category = request.form.get('category', '').strip()
        amt = float(request.form.get('amt', 0))
        gender = request.form.get('gender', '').strip()
        city = request.form.get('city', '').strip()
        state = request.form.get('state', '').strip()
        city_pop = int(request.form.get('city_pop', 0))
        job = request.form.get('job', '').strip()
        merch_lat = float(request.form.get('merch_lat', 0.0))
        merch_long = float(request.form.get('merch_long', 0.0))

        features = pd.DataFrame([{
            "category": category,
            "amt": amt,
            "gender": gender,
            "city": city,
            "state": state,
            "city_pop": city_pop,
            "job": job,
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }])

        prediction = int(model.predict(features).tolist()[0])
        result_text = "❌ Transaction frauduleuse détectée !" if prediction == 1 else "✅ Transaction normale."

        tx = Transaction(
            user_id=current_user.id,
            category=category or None,
            amt=amt,
            gender=gender or None,
            city=city or None,
            state=state or None,
            job=job or None,
            city_pop=city_pop,
            merch_lat=merch_lat,
            merch_long=merch_long,
            prediction=prediction
        )
        db.session.add(tx)
        db.session.commit()

        last_tx = Transaction.query.filter_by(user_id=current_user.id)\
                                   .order_by(Transaction.created_at.desc())\
                                   .limit(12).all()
        
        return render_template('index.html', 
                             prediction=result_text, 
                             history=last_tx, 
                             username=current_user.username, 
                             cities=CITIES, 
                             states=STATES, 
                             jobs=JOBS)

    except Exception as e:
        flash(f"Erreur lors de la prédiction: {str(e)}", "danger")
        return redirect(url_for('home'))

@app.route('/history')
@login_required
def history():
    all_tx = Transaction.query.filter_by(user_id=current_user.id)\
                              .order_by(Transaction.created_at.desc()).all()
    return render_template('history.html', history=all_tx)


# ======== DASHBOARD PERSONNEL (ALL PREDICTIONS FOR LOGGED-IN USER) ========
@app.route('/dashboard')
@login_required
def dashboard():
    """Dashboard showing ALL predictions made by the logged-in user"""
    
    # ===== GLOBAL STATS (ALL USERS) FOR THE STAT CARDS =====
    all_transactions = Transaction.query.all()
    total_transactions_global = len(all_transactions)
    total_frauds_global = sum(1 for tx in all_transactions if tx.prediction == 1)
    total_normal_global = sum(1 for tx in all_transactions if tx.prediction == 0)
    fraud_percentage_global = (total_frauds_global / total_transactions_global * 100) if total_transactions_global > 0 else 0
    total_users = db.session.query(User).count()
    
    # ===== PERSONAL GRAPHS (ONLY CURRENT USER'S PREDICTIONS) =====
    user_transactions = Transaction.query.filter_by(user_id=current_user.id)\
                                         .order_by(Transaction.created_at.desc()).all()
    
    if not user_transactions:
        flash("You haven't made any predictions yet. Make some predictions first to see your dashboard!", "info")
        return render_template('dashboard.html',
                             total_transactions=total_transactions_global,
                             total_frauds=total_frauds_global,
                             total_normal=total_normal_global,
                             fraud_percentage=fraud_percentage_global,
                             total_users=total_users,
                             pie_html=None,
                             time_html=None,
                             gender_html=None,
                             username=current_user.username)
    
    # Convert user's transactions to DataFrame
    df = pd.DataFrame([{
        "category": tx.category,
        "amt": tx.amt,
        "gender": tx.gender,
        "city": tx.city,
        "state": tx.state,
        "job": tx.job,
        "city_pop": tx.city_pop,
        "merch_lat": tx.merch_lat,
        "merch_long": tx.merch_long,
        "prediction": tx.prediction,
        "created_at": tx.created_at
    } for tx in user_transactions])
    
    # GRAPH 1: Pie Chart - Fraud vs Normal (MY predictions)
    fraud_counts = df['prediction'].value_counts().sort_index()
    labels = ["✅ Normal" if i == 0 else "❌ Fraude" for i in fraud_counts.index]
    colors = ['#4caf50', '#f44336']
    pie_fig = px.pie(values=fraud_counts.values, names=labels, 
                     title="My Predictions: Fraud vs Normal",
                     color_discrete_sequence=colors)
    pie_html = pie_fig.to_html(full_html=False)
    
    # GRAPH 2: Bar Chart - Transactions by Gender (MY predictions)
    if 'gender' in df.columns and df['gender'].notna().any():
        gender_stats = df.groupby(['gender', 'prediction']).size().reset_index(name='count')
        gender_stats['label'] = gender_stats['prediction'].map({0: 'Normal', 1: 'Fraude'})
        gender_fig = px.bar(gender_stats, x='gender', y='count', color='label',
                            title="My Transactions by Gender",
                            labels={'gender': 'Gender', 'count': 'Count', 'label': 'Type'},
                            barmode='group',
                            color_discrete_map={'Normal': '#4caf50', 'Fraude': '#f44336'})
        gender_html = gender_fig.to_html(full_html=False)
    else:
        gender_html = None
    
    # GRAPH 3: LINE CHART - Count of predictions over time
    df_sorted = df.sort_values('created_at').reset_index(drop=True)
    df_sorted['datetime'] = pd.to_datetime(df_sorted['created_at'])
    df_sorted['label'] = df_sorted['prediction'].map({0: 'Normal', 1: 'Fraude'})
    
    # Group by datetime and count predictions
    time_counts = df_sorted.groupby(['datetime', 'label']).size().reset_index(name='count')
    
    # Create cumulative count for each type
    time_counts_sorted = time_counts.sort_values('datetime')
    time_counts_sorted['cumulative_count'] = time_counts_sorted.groupby('label')['count'].cumsum()
    
    # Create LINE chart with COUNT on Y-axis
    time_fig = px.line(time_counts_sorted, 
                       x='datetime', 
                       y='cumulative_count',
                       color='label',
                       title="Number of My Predictions Over Time",
                       labels={'datetime': 'Date and Time', 
                              'cumulative_count': 'Number of Predictions', 
                              'label': 'Type'},
                       color_discrete_map={'Normal': '#4caf50', 'Fraude': '#f44336'},
                       markers=True)
    
    # Customize appearance
    time_fig.update_traces(line=dict(width=3), marker=dict(size=10))
    time_fig.update_xaxes(
        title_text="Date and Time",
        tickformat="%Y-%m-%d\n%H:%M"
    )
    time_fig.update_yaxes(
        title_text="Number of Predictions",
        dtick=1  # Show every integer (1, 2, 3, 4...)
    )
    time_fig.update_layout(
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    time_html = time_fig.to_html(full_html=False)
    
    return render_template('dashboard.html',
                         total_transactions=total_transactions_global,
                         total_frauds=total_frauds_global,
                         total_normal=total_normal_global,
                         fraud_percentage=fraud_percentage_global,
                         total_users=total_users,
                         pie_html=pie_html,
                         time_html=time_html,
                         gender_html=gender_html,
                         username=current_user.username)

# ======== PAGE D'ANALYSE GLOBALE (CSV) ========
@app.route('/analysis', methods=['GET', 'POST'])
@login_required
def analysis():
    graph_html = None
    graph_type = None

    if not os.path.exists(CSV_PATH):
        flash("Fichier CSV introuvable!", "danger")
        return redirect(url_for('home'))

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        flash(f"Erreur lecture CSV: {e}", "danger")
        return redirect(url_for('home'))

    df_cols = ["category", "amt", "gender", "city", "state", "city_pop", "job", "merch_lat", "merch_long"]

    if request.method == 'POST':
        analysis_type = request.form.get('analysis_type')
        cols = [c for c in [request.form.get('col1'), request.form.get('col2'), request.form.get('col3')] if c]
        graph_type = request.form.get('graph_type')

        fig = None

        if analysis_type == 'univarie' and len(cols) == 1:
            col = cols[0]
            if graph_type == 'histogram':
                fig = px.histogram(df, x=col, title=f'Histogramme de {col}')
            elif graph_type == 'box':
                fig = px.box(df, y=col, title=f'Boxplot de {col}')
            elif graph_type == 'pie':
                fig = px.pie(df, names=col, title=f'Camembert de {col}')
            elif graph_type == 'line':
                fig = px.line(df, y=col, title=f'Ligne de {col}')

        elif analysis_type == 'bivarie' and len(cols) == 2:
            x, y = cols[0], cols[1]
            if graph_type == 'scatter':
                fig = px.scatter(df, x=x, y=y, title=f'Scatter {x} vs {y}')
            elif graph_type == 'box':
                fig = px.box(df, x=x, y=y, title=f'Boxplot {x} vs {y}')
            elif graph_type == 'bar':
                fig = px.bar(df, x=x, y=y, title=f'Barres {x} vs {y}')
            elif graph_type == 'line':
                fig = px.line(df, x=x, y=y, title=f'Ligne {x} vs {y}')
            elif graph_type == 'heatmap':
                corr = df[[x, y]].corr()
                fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(),
                                                  y=corr.columns.tolist(), colorscale='Viridis')
                fig.update_layout(title=f'Heatmap de corrélation entre {x} et {y}')
            elif graph_type == 'correlation':
                corr_value = df[x].corr(df[y])
                flash(f"Corrélation entre {x} et {y} : {corr_value:.3f}", "info")
                fig = px.scatter(df, x=x, y=y, trendline="ols",
                                 title=f"Corrélation {x} vs {y} (r={corr_value:.2f})")

        elif analysis_type == 'trivarie' and len(cols) == 3:
            fig = px.scatter_3d(df, x=cols[0], y=cols[1], z=cols[2],
                                title=f'Scatter 3D : {cols[0]}, {cols[1]}, {cols[2]}')

        else:
            flash("⚠ Choix invalide. Vérifie ton type d'analyse et tes colonnes.", "danger")

        if fig:
            fig.update_layout(template="plotly_white")
            graph_html = fig.to_html(full_html=False)

    return render_template('analysis.html', df_cols=df_cols, graph_html=graph_html,
                           graph_type=graph_type, username=current_user.username)

# ======== ANALYSE DES PREDICTIONS D'UN UTILISATEUR ========
@app.route('/user_analysis', methods=['GET', 'POST'])
@login_required
def user_analysis():
    user_tx = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.created_at).all()

    if not user_tx:
        return render_template('user_analysis.html',
                               pie_html=None,
                               box_html=None,
                               custom_html=None,
                               user_cols=["category", "amt", "gender", "city", "state", "job", "city_pop", "merch_lat", "merch_long"],
                               username=current_user.username,
                               info_message="Aucune transaction enregistrée pour ton compte — effectue une prédiction d'abord pour voir l'analyse.")

    df = pd.DataFrame([{
        "category": tx.category,
        "amt": tx.amt,
        "gender": tx.gender,
        "city": tx.city,
        "state": tx.state,
        "job": tx.job,
        "city_pop": tx.city_pop,
        "merch_lat": tx.merch_lat,
        "merch_long": tx.merch_long,
        "prediction": tx.prediction,
        "created_at": tx.created_at
    } for tx in user_tx])

    user_cols = ["category", "amt", "gender", "city", "state", "job", "city_pop", "merch_lat", "merch_long"]

    fraud_count = df['prediction'].value_counts().sort_index()
    labels = ["Normal" if i == 0 else "Fraude" for i in fraud_count.index]
    pie_fig = px.pie(values=fraud_count.values, names=labels, title="Répartition des Prédictions (0=Normal, 1=Fraude)")
    pie_html = pie_fig.to_html(full_html=False)

    box_fig = px.box(df, x='prediction', y='amt', title='Distribution des montants par prediction', labels={'prediction': 'Prediction (0=Normal,1=Fraude)'})
    box_html = box_fig.to_html(full_html=False)

    custom_html = None
    if request.method == 'POST':
        sel_type = request.form.get('user_analysis_type')
        sel_cols = [c for c in [request.form.get('u_col1'), request.form.get('u_col2'), request.form.get('u_col3')] if c]
        sel_graph = request.form.get('u_graph_type')
        fig = None
        try:
            if sel_type == 'univarie' and len(sel_cols) == 1:
                c = sel_cols[0]
                if sel_graph == 'histogram':
                    fig = px.histogram(df, x=c, title=f'Histogramme de {c}')
                elif sel_graph == 'box':
                    fig = px.box(df, y=c, title=f'Boxplot de {c}')
                elif sel_graph == 'pie':
                    fig = px.pie(df, names=c, title=f'Camembert de {c}')
                elif sel_graph == 'line':
                    if c == 'amt':
                        df_sorted = df.sort_values('created_at')
                        fig = px.line(df_sorted, x='created_at', y='amt', title='Montant dans le temps')
                    else:
                        fig = px.line(df, y=c, title=f'Ligne de {c}')
            elif sel_type == 'bivarie' and len(sel_cols) == 2:
                x, y = sel_cols
                if sel_graph == 'scatter':
                    fig = px.scatter(df, x=x, y=y, color='prediction', title=f'Scatter {x} vs {y}')
                elif sel_graph == 'bar':
                    fig = px.bar(df, x=x, y=y, color='prediction', title=f'Barres {x} vs {y}')
                elif sel_graph == 'heatmap':
                    corr = pd.DataFrame({x: df[x], y: df[y]}).corr()
                    fig = ff.create_annotated_heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist())
                elif sel_graph == 'line':
                    fig = px.line(df, x=x, y=y, title=f'Ligne {x} vs {y}')
            elif sel_type == 'trivarie' and len(sel_cols) == 3:
                fig = px.scatter_3d(df, x=sel_cols[0], y=sel_cols[1], z=sel_cols[2], color='prediction',
                                    title=f'Scatter 3D : {sel_cols}')
            else:
                flash("Choix invalide pour l'analyse utilisateur.", "danger")
            if fig:
                fig.update_layout(template="plotly_white")
                custom_html = fig.to_html(full_html=False)
        except Exception as e:
            flash(f"Erreur génération graphe: {e}", "danger")

    return render_template('user_analysis.html',
                           pie_html=pie_html,
                           box_html=box_html,
                           custom_html=custom_html,
                           user_cols=user_cols,
                           username=current_user.username)

# ======== RUN ========
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)