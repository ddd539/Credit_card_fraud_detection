import os
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
import pickle
import pandas as pd
import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import plotly.express as px
import plotly.figure_factory as ff
import requests

# ======== PAGE CONFIG ========
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== FILE DOWNLOAD FROM GOOGLE DRIVE ========
def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive"""
    if os.path.exists(destination):
        return True
    
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large files with confirmation token
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
    
    # Save file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    return os.path.exists(destination)

# ======== GOOGLE DRIVE FILE IDs ========
# TO GET FILE ID:
# 1. Upload your file to Google Drive
# 2. Right-click → Get shareable link
# 3. Make sure "Anyone with the link can view" is enabled
# 4. Extract the ID from the URL: https://drive.google.com/file/d/FILE_ID_HERE/view
# 5. Paste the FILE_ID below

MODEL_FILE_ID = "1ZkmGBiDrgHO-UpCJ1r0jR1lR6Hzo4I1c"  # Your model file
CSV_FILE_ID = "1e1oB9sGsdI7YtBUo_VnL4-S-fe5hEKJw"  # Your CSV file

MODEL_PATH = "rf_pipeline2.pkl"
CSV_PATH = "fraudTrain.csv"

# Download files on startup
@st.cache_resource
def download_model():
    """Download model from Google Drive"""
    with st.spinner("📥 Downloading ML model from Google Drive..."):
        try:
            success = download_file_from_google_drive(MODEL_FILE_ID, MODEL_PATH)
            if success:
                st.success("✅ Model downloaded successfully!")
                return True
            else:
                st.error("❌ Failed to download model")
                return False
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            return False

@st.cache_resource
def download_csv():
    """Download CSV from Google Drive"""
    with st.spinner("📥 Downloading training data from Google Drive..."):
        try:
            success = download_file_from_google_drive(CSV_FILE_ID, CSV_PATH)
            if success:
                st.success("✅ CSV downloaded successfully!")
                return True
            else:
                st.error("❌ Failed to download CSV")
                return False
        except Exception as e:
            st.error(f"❌ Error downloading CSV: {e}")
            return False

# Download files if they don't exist
if MODEL_FILE_ID != "YOUR_MODEL_FILE_ID_HERE":
    download_model()
else:
    st.warning("⚠️ Please add your Google Drive file IDs in app.py")

if CSV_FILE_ID != "YOUR_CSV_FILE_ID_HERE":
    download_csv()

# ======== DATABASE SETUP ========
try:
    DATABASE_URI = st.secrets["database"]["uri"]
except:
    DATABASE_URI = 'sqlite:///fraud_db.sqlite'
    st.warning("⚠️ Using SQLite database. Configure MySQL in Streamlit secrets for production.")

Base = declarative_base()
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)

# ======== MODELS ========
class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    transactions = relationship('Transaction', back_populates='user')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Transaction(Base):
    __tablename__ = 'transactions'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.id'), nullable=False)
    category = Column(String(100))
    amt = Column(Float)
    gender = Column(String(20))
    city = Column(String(100))
    state = Column(String(100))
    job = Column(String(100))
    city_pop = Column(Integer)
    merch_lat = Column(Float)
    merch_long = Column(Float)
    prediction = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship('User', back_populates='transactions')

Base.metadata.create_all(engine)

# ======== SESSION STATE INITIALIZATION ========
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None

# ======== LOAD MODEL ========
@st.cache_resource
def load_model():
    """Load the ML model"""
    try:
        if os.path.exists(MODEL_PATH):
            model = pickle.load(open(MODEL_PATH, "rb"))
            return model
        else:
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None
    except Exception as e:
        st.error(f"❌ Unable to load model: {e}")
        return None

model = load_model()

# ======== LOAD UNIQUE VALUES ========
@st.cache_data
def load_unique_values():
    """Load unique values from CSV once at startup"""
    try:
        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            cities = sorted(df['city'].dropna().unique().tolist())
            states = sorted(df['state'].dropna().unique().tolist())
            jobs = sorted(df['job'].dropna().unique().tolist())
            return cities, states, jobs
        else:
            st.warning(f"⚠️ CSV file not found: {CSV_PATH}")
            return [], [], []
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return [], [], []

CITIES, STATES, JOBS = load_unique_values()

# ======== AUTHENTICATION FUNCTIONS ========
def register_user(username, password):
    """Register a new user"""
    session = Session()
    try:
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            return False, "Username already taken"
        
        new_user = User(username=username)
        new_user.set_password(password)
        session.add(new_user)
        session.commit()
        return True, "Registration successful"
    except Exception as e:
        session.rollback()
        return False, f"Error: {e}"
    finally:
        session.close()

def login_user(username, password):
    """Login a user"""
    session = Session()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and user.check_password(password):
            st.session_state.logged_in = True
            st.session_state.user_id = user.id
            st.session_state.username = user.username
            return True, "Login successful"
        return False, "Invalid username or password"
    finally:
        session.close()

def logout_user():
    """Logout the current user"""
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# ======== MAIN APP ========
def main():
    if not st.session_state.logged_in:
        show_auth_page()
    else:
        show_main_app()

def show_auth_page():
    """Show login/registration page"""
    st.title("🔒 Fraud Detection System")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username and password:
                    success, message = login_user(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please fill in all fields")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("Username", key="reg_username")
            new_password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register = st.form_submit_button("Register")
            
            if register:
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("Passwords don't match")
                    else:
                        success, message = register_user(new_username, new_password)
                        if success:
                            st.success(message)
                            st.info("Please login with your credentials")
                        else:
                            st.error(message)
                else:
                    st.warning("Please fill in all fields")

def show_main_app():
    """Show main application interface"""
    st.sidebar.title(f"👤 {st.session_state.username}")
    
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Dashboard", "📈 Analysis", "👤 User Analysis", "📜 History"]
    )
    
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Dashboard":
        show_dashboard_page()
    elif page == "📈 Analysis":
        show_analysis_page()
    elif page == "👤 User Analysis":
        show_user_analysis_page()
    elif page == "📜 History":
        show_history_page()

def show_home_page():
    """Home page with prediction form"""
    st.title("🏠 Fraud Detection - Make Prediction")
    
    st.markdown("### Enter Transaction Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        category = st.text_input("Category", placeholder="e.g., grocery_pos")
        amt = st.number_input("Amount ($)", min_value=0.0, step=0.01)
        gender = st.selectbox("Gender", ["M", "F"])
    
    with col2:
        city = st.selectbox("City", [""] + CITIES)
        state = st.selectbox("State", [""] + STATES)
        city_pop = st.number_input("City Population", min_value=0, step=1)
    
    with col3:
        job = st.selectbox("Job", [""] + JOBS)
        merch_lat = st.number_input("Merchant Latitude", value=0.0, format="%.6f")
        merch_long = st.number_input("Merchant Longitude", value=0.0, format="%.6f")
    
    if st.button("🔍 Predict", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded!")
            return
        
        try:
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
            
            prediction = int(model.predict(features)[0])
            
            session = Session()
            try:
                tx = Transaction(
                    user_id=st.session_state.user_id,
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
                session.add(tx)
                session.commit()
            finally:
                session.close()
            
            st.markdown("---")
            if prediction == 1:
                st.error("❌ **FRAUD DETECTED!** This transaction appears to be fraudulent.")
            else:
                st.success("✅ **NORMAL TRANSACTION** This transaction appears legitimate.")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    
    st.markdown("---")
    st.subheader("📋 Recent Predictions")
    
    session = Session()
    try:
        recent_tx = session.query(Transaction)\
            .filter_by(user_id=st.session_state.user_id)\
            .order_by(Transaction.created_at.desc())\
            .limit(10).all()
        
        if recent_tx:
            data = []
            for tx in recent_tx:
                data.append({
                    "Date": tx.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Category": tx.category,
                    "Amount": f"${tx.amt:.2f}",
                    "City": tx.city,
                    "State": tx.state,
                    "Result": "❌ Fraud" if tx.prediction == 1 else "✅ Normal"
                })
            st.dataframe(data, use_container_width=True)
        else:
            st.info("No predictions yet. Make your first prediction above!")
    finally:
        session.close()

def show_dashboard_page():
    """Dashboard showing statistics"""
    st.title("📊 Dashboard")
    
    session = Session()
    try:
        all_tx = session.query(Transaction).all()
        user_tx = session.query(Transaction)\
            .filter_by(user_id=st.session_state.user_id).all()
        
        total_users = session.query(User).count()
        
        total_global = len(all_tx)
        fraud_global = sum(1 for tx in all_tx if tx.prediction == 1)
        normal_global = total_global - fraud_global
        fraud_pct = (fraud_global / total_global * 100) if total_global > 0 else 0
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Transactions", total_global)
        with col2:
            st.metric("Fraudulent", fraud_global)
        with col3:
            st.metric("Normal", normal_global)
        with col4:
            st.metric("Fraud Rate", f"{fraud_pct:.1f}%")
        with col5:
            st.metric("Total Users", total_users)
        
        if not user_tx:
            st.info("You haven't made any predictions yet. Make some predictions first!")
            return
        
        df = pd.DataFrame([{
            "category": tx.category,
            "amt": tx.amt,
            "gender": tx.gender,
            "city": tx.city,
            "state": tx.state,
            "job": tx.job,
            "prediction": tx.prediction,
            "created_at": tx.created_at
        } for tx in user_tx])
        
        st.markdown("---")
        st.subheader("📈 My Predictions Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fraud_counts = df['prediction'].value_counts().sort_index()
            labels = ["✅ Normal" if i == 0 else "❌ Fraud" for i in fraud_counts.index]
            colors = ['#4caf50', '#f44336']
            pie_fig = px.pie(
                values=fraud_counts.values,
                names=labels,
                title="My Predictions: Fraud vs Normal",
                color_discrete_sequence=colors
            )
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            if 'gender' in df.columns and df['gender'].notna().any():
                gender_stats = df.groupby(['gender', 'prediction']).size().reset_index(name='count')
                gender_stats['label'] = gender_stats['prediction'].map({0: 'Normal', 1: 'Fraud'})
                gender_fig = px.bar(
                    gender_stats,
                    x='gender',
                    y='count',
                    color='label',
                    title="My Transactions by Gender",
                    barmode='group',
                    color_discrete_map={'Normal': '#4caf50', 'Fraud': '#f44336'}
                )
                st.plotly_chart(gender_fig, use_container_width=True)
        
        st.markdown("---")
        df_sorted = df.sort_values('created_at').reset_index(drop=True)
        df_sorted['datetime'] = pd.to_datetime(df_sorted['created_at'])
        df_sorted['label'] = df_sorted['prediction'].map({0: 'Normal', 1: 'Fraud'})
        
        time_counts = df_sorted.groupby(['datetime', 'label']).size().reset_index(name='count')
        time_counts_sorted = time_counts.sort_values('datetime')
        time_counts_sorted['cumulative_count'] = time_counts_sorted.groupby('label')['count'].cumsum()
        
        time_fig = px.line(
            time_counts_sorted,
            x='datetime',
            y='cumulative_count',
            color='label',
            title="Number of My Predictions Over Time",
            color_discrete_map={'Normal': '#4caf50', 'Fraud': '#f44336'},
            markers=True
        )
        time_fig.update_traces(line=dict(width=3), marker=dict(size=10))
        st.plotly_chart(time_fig, use_container_width=True)
        
    finally:
        session.close()

def show_analysis_page():
    """Global CSV analysis page"""
    st.title("📈 Global Analysis (CSV Data)")
    
    if not os.path.exists(CSV_PATH):
        st.error("CSV file not found!")
        return
    
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return
    
    df_cols = ["category", "amt", "gender", "city", "state", "city_pop", "job", "merch_lat", "merch_long"]
    
    st.subheader("Select Analysis Type")
    
    analysis_type = st.radio("Analysis Type", ["Univariate", "Bivariate", "Trivariate"])
    
    if analysis_type == "Univariate":
        col1, col2 = st.columns([2, 1])
        with col1:
            column = st.selectbox("Select Column", df_cols)
        with col2:
            graph_type = st.selectbox("Graph Type", ["histogram", "box", "pie", "line"])
        
        if st.button("Generate Graph"):
            if graph_type == "histogram":
                fig = px.histogram(df, x=column, title=f'Histogram of {column}')
            elif graph_type == "box":
                fig = px.box(df, y=column, title=f'Boxplot of {column}')
            elif graph_type == "pie":
                fig = px.pie(df, names=column, title=f'Pie Chart of {column}')
            elif graph_type == "line":
                fig = px.line(df, y=column, title=f'Line Chart of {column}')
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Bivariate":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            x_col = st.selectbox("X Axis", df_cols)
        with col2:
            y_col = st.selectbox("Y Axis", df_cols)
        with col3:
            graph_type = st.selectbox("Graph Type", ["scatter", "box", "bar", "line", "heatmap", "correlation"])
        
        if st.button("Generate Graph"):
            if graph_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter {x_col} vs {y_col}')
            elif graph_type == "box":
                fig = px.box(df, x=x_col, y=y_col, title=f'Boxplot {x_col} vs {y_col}')
            elif graph_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col, title=f'Bar Chart {x_col} vs {y_col}')
            elif graph_type == "line":
                fig = px.line(df, x=x_col, y=y_col, title=f'Line Chart {x_col} vs {y_col}')
            elif graph_type == "heatmap":
                corr = df[[x_col, y_col]].corr()
                fig = ff.create_annotated_heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.columns.tolist(),
                    colorscale='Viridis'
                )
                fig.update_layout(title=f'Correlation Heatmap: {x_col} vs {y_col}')
            elif graph_type == "correlation":
                corr_value = df[x_col].corr(df[y_col])
                st.info(f"Correlation between {x_col} and {y_col}: {corr_value:.3f}")
                fig = px.scatter(df, x=x_col, y=y_col, trendline="ols", 
                               title=f"Correlation {x_col} vs {y_col} (r={corr_value:.2f})")
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            x_col = st.selectbox("X Axis", df_cols)
        with col2:
            y_col = st.selectbox("Y Axis", df_cols)
        with col3:
            z_col = st.selectbox("Z Axis", df_cols)
        
        if st.button("Generate 3D Scatter"):
            fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col,
                              title=f'3D Scatter: {x_col}, {y_col}, {z_col}')
            st.plotly_chart(fig, use_container_width=True)

def show_user_analysis_page():
    """User-specific analysis page"""
    st.title("👤 My Predictions Analysis")
    
    session = Session()
    try:
        user_tx = session.query(Transaction)\
            .filter_by(user_id=st.session_state.user_id)\
            .order_by(Transaction.created_at).all()
        
        if not user_tx:
            st.info("No transactions recorded. Make a prediction first to see analysis.")
            return
        
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            fraud_count = df['prediction'].value_counts().sort_index()
            labels = ["Normal" if i == 0 else "Fraud" for i in fraud_count.index]
            pie_fig = px.pie(values=fraud_count.values, names=labels,
                           title="Distribution of Predictions")
            st.plotly_chart(pie_fig, use_container_width=True)
        
        with col2:
            box_fig = px.box(df, x='prediction', y='amt',
                           title='Amount Distribution by Prediction',
                           labels={'prediction': 'Prediction (0=Normal, 1=Fraud)'})
            st.plotly_chart(box_fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Custom Analysis")
        
        analysis_type = st.radio("Analysis Type", ["Univariate", "Bivariate", "Trivariate"], key="user_analysis")
        
        if analysis_type == "Univariate":
            col1, col2 = st.columns([2, 1])
            with col1:
                column = st.selectbox("Select Column", user_cols)
            with col2:
                graph_type = st.selectbox("Graph Type", ["histogram", "box", "pie", "line"])
            
            if st.button("Generate"):
                if graph_type == "histogram":
                    fig = px.histogram(df, x=column, title=f'Histogram of {column}')
                elif graph_type == "box":
                    fig = px.box(df, y=column, title=f'Boxplot of {column}')
                elif graph_type == "pie":
                    fig = px.pie(df, names=column, title=f'Pie Chart of {column}')
                elif graph_type == "line":
                    if column == 'amt':
                        df_sorted = df.sort_values('created_at')
                        fig = px.line(df_sorted, x='created_at', y='amt', title='Amount Over Time')
                    else:
                        fig = px.line(df, y=column, title=f'Line Chart of {column}')
                
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Bivariate":
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                x_col = st.selectbox("X Axis", user_cols, key="biv_x")
            with col2:
                y_col = st.selectbox("Y Axis", user_cols, key="biv_y")
            with col3:
                graph_type = st.selectbox("Graph Type", ["scatter", "bar", "heatmap", "line"])
            
            if st.button("Generate"):
                if graph_type == "scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, color='prediction',
                                   title=f'Scatter {x_col} vs {y_col}')
                elif graph_type == "bar":
                    fig = px.bar(df, x=x_col, y=y_col, color='prediction',
                               title=f'Bar Chart {x_col} vs {y_col}')
                elif graph_type == "heatmap":
                    corr = pd.DataFrame({x_col: df[x_col], y_col: df[y_col]}).corr()
                    fig = ff.create_annotated_heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.columns.tolist()
                    )
                elif graph_type == "line":
                    fig = px.line(df, x=x_col, y=y_col, title=f'Line Chart {x_col} vs {y_col}')
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X Axis", user_cols, key="tri_x")
            with col2:
                y_col = st.selectbox("Y Axis", user_cols, key="tri_y")
            with col3:
                z_col = st.selectbox("Z Axis", user_cols, key="tri_z")
            
            if st.button("Generate 3D"):
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color='prediction',
                                  title=f'3D Scatter: {x_col}, {y_col}, {z_col}')
                st.plotly_chart(fig, use_container_width=True)
        
    finally:
        session.close()

def show_history_page():
    """Show all user transactions"""
    st.title("📜 Transaction History")
    
    session = Session()
    try:
        all_tx = session.query(Transaction)\
            .filter_by(user_id=st.session_state.user_id)\
            .order_by(Transaction.created_at.desc()).all()
        
        if not all_tx:
            st.info("No transaction history yet.")
            return
        
        data = []
        for tx in all_tx:
            data.append({
                "ID": tx.id,
                "Date": tx.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "Category": tx.category,
                "Amount": f"${tx.amt:.2f}",
                "Gender": tx.gender,
                "City": tx.city,
                "State": tx.state,
                "Job": tx.job,
                "City Pop": tx.city_pop,
                "Lat": tx.merch_lat,
                "Long": tx.merch_long,
                "Result": "❌ Fraud" if tx.prediction == 1 else "✅ Normal"
            })
        
        df_display = pd.DataFrame(data)
        st.dataframe(df_display, use_container_width=True, height=600)
        
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download History as CSV",
            data=csv,
            file_name=f"transaction_history_{st.session_state.username}.csv",
            mime="text/csv"
        )
        
    finally:
        session.close()

if __name__ == "__main__":
    main()