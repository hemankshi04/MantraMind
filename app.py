"""
============================================================
  app.py  -  Mantra & Mental Health Flask Web App

  HOW TO RUN:
    python app.py
  THEN OPEN:
    http://localhost:5000

  REQUIRES:
    model/best_dl_model.pkl  (run train_dl_enhanced_model.py first)
============================================================
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import pandas as pd, numpy as np, joblib, os, json, random
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'mantra_wellness_2024'
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'wellness.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── DB Models ─────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id         = db.Column(db.Integer, primary_key=True)
    name       = db.Column(db.String(100), nullable=False)
    email      = db.Column(db.String(150), unique=True, nullable=False)
    password   = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    responses  = db.relationship('Response', backref='user', lazy=True)

class Response(db.Model):
    __tablename__ = 'responses'
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    struggles      = db.Column(db.String(300))
    sleep_hours    = db.Column(db.Float)
    stress_level   = db.Column(db.Integer)
    exercise_freq  = db.Column(db.String(50))
    mood           = db.Column(db.String(30))
    wellness_score = db.Column(db.Float)
    score_category = db.Column(db.String(20))
    timestamp      = db.Column(db.DateTime, default=datetime.utcnow)

# ── Load DL Model ─────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, 'model')
try:
    dl_scaler = joblib.load(os.path.join(MODEL_DIR, 'dl_scaler.pkl'))
    le_mantra = joblib.load(os.path.join(MODEL_DIR, 'le_mantra.pkl'))
    le_gender = joblib.load(os.path.join(MODEL_DIR, 'le_gender.pkl'))
    le_exp = joblib.load(os.path.join(MODEL_DIR, 'le_exp.pkl'))
    le_target = joblib.load(os.path.join(MODEL_DIR, 'le_target.pkl'))
    FEATURE_COLS = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

    # Try TensorFlow first, then sklearn DL model
    try:
        from tensorflow import keras
        dl_model = keras.models.load_model(os.path.join(MODEL_DIR, 'best_dl_model.h5'))
        MODEL_TYPE = 'tensorflow'
        print("TensorFlow DL Model loaded")
    except:
        dl_model = joblib.load(os.path.join(MODEL_DIR, 'best_dl_model.pkl'))
        MODEL_TYPE = 'sklearn_dl'
        print("sklearn DL (MLP) Model loaded")
except:
    print("WARNING: DL Model not found. Run train_dl_enhanced_model.py first!")
    dl_model = None

# ── Load ML Fallback Model ─────────────────────────────────────────────────
try:
    ml_model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    print("ML Fallback Model (GradientBoosting) loaded")
except:
    print("WARNING: ML fallback model not found!")
    ml_model = None

# ── Model Performance Data (from training results) ─────────────────────────
MODEL_COMPARISON = {
    "DL_MLP": {
        "name": "Deep Learning (MLP Neural Network)",
        "type": "DL",
        "accuracy": 0.8562,
        "f1_score": 0.8582,
        "mse": 0.2938,
        "dataset": "All Combined (800)",
        "color": "#4FC3F7",
        "icon": "🧠"
    },
    "ML_GB": {
        "name": "ML Gradient Boosting",
        "type": "ML",
        "accuracy": 0.9375,
        "f1_score": 0.9375,
        "mse": 0.1437,
        "dataset": "All Combined (800)",
        "color": "#3EE07A",
        "icon": "🌲"
    },
    "ML_RF": {
        "name": "ML Random Forest",
        "type": "ML",
        "accuracy": 0.9313,
        "f1_score": 0.9300,
        "mse": 0.1437,
        "dataset": "All Combined (800)",
        "color": "#B39DDB",
        "icon": "🌳"
    },
    "ML_LR": {
        "name": "ML Logistic Regression",
        "type": "ML",
        "accuracy": 0.9125,
        "f1_score": 0.9134,
        "mse": 0.1625,
        "dataset": "All Combined (800)",
        "color": "#FF8C42",
        "icon": "📊"
    }
}

# ── Auth decorator ────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Wellness scorer ───────────────────────────────────────────
def calc_wellness(sleep, stress, exercise, mood, struggles):
    sleep_pts = 30 if 7<=sleep<=9 else 22 if 6<=sleep<7 else 14 if 5<=sleep<6 else 6
    stress_pts = round(30 - (stress-1)*(27/9), 1)
    ex_map = {'daily':20,'3-4':15,'1-2':10,'rarely':5,'never':0}
    ex_pts = ex_map.get(exercise, 5)
    mood_pts = {'happy':15,'neutral':8,'sad':3}.get(mood, 8)
    str_pts = max(0, 5 - len(struggles))
    score = round(min(100, sleep_pts+stress_pts+ex_pts+mood_pts+str_pts), 1)
    cat   = 'excellent' if score>=80 else 'good' if score>=60 else 'fair' if score>=40 else 'needs-care'
    return score, cat

AFFIRMATIONS = [
    "You are stronger than you think.",
    "Today I choose peace over anxiety.",
    "My mind is calm. My body is healthy.",
    "I release what I cannot control.",
    "Every breath fills me with clarity.",
    "I am worthy of rest and happiness.",
    "Small steps forward are still progress.",
    "This moment is temporary. My strength is permanent.",
]

# ── ROUTES ───────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect(url_for('dashboard') if 'user_id' in session else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if 'user_id' in session: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form['email'].strip().lower()).first()
        if user and check_password_hash(user.password, request.form['password']):
            session['user_id']   = user.id
            session['user_name'] = user.name
            flash(f'Welcome back, {user.name}! 🙏', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        name, email = request.form['name'].strip(), request.form['email'].strip().lower()
        pwd, conf   = request.form['password'], request.form['confirm_password']
        if not name or not email or not pwd:
            flash('All fields required.', 'error')
        elif pwd != conf:
            flash('Passwords do not match.', 'error')
        elif len(pwd) < 6:
            flash('Password must be 6+ characters.', 'error')
        elif User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
        else:
            u = User(name=name, email=email, password=generate_password_hash(pwd))
            db.session.add(u); db.session.commit()
            session['user_id'] = u.id; session['user_name'] = u.name
            flash(f'Welcome, {name}! 🌟', 'success')
            return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out. See you soon! 🙏', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    history = Response.query.filter_by(user_id=user.id).order_by(Response.timestamp.desc()).limit(7).all()
    latest  = history[0] if history else None
    chart_labels = json.dumps([r.timestamp.strftime('%d %b') for r in reversed(history)])
    chart_scores = json.dumps([r.wellness_score for r in reversed(history)])
    return render_template('dashboard.html',
        user=user, latest=latest, history=history,
        chart_labels=chart_labels, chart_scores=chart_scores,
        affirmation=random.choice(AFFIRMATIONS),
        total_checkins=len(history))

@app.route('/checkin', methods=['GET','POST'])
@login_required
def checkin():
    if request.method == 'GET':
        return render_template('checkin.html')
    struggles = request.form.getlist('struggles')
    sleep     = float(request.form.get('sleep_hours', 7))
    stress    = int(request.form.get('stress_level', 5))
    exercise  = request.form.get('exercise_freq', 'rarely')
    mood      = request.form.get('mood', 'neutral')
    score, cat = calc_wellness(sleep, stress, exercise, mood, struggles)
    r = Response(user_id=session['user_id'], struggles=json.dumps(struggles),
                 sleep_hours=sleep, stress_level=stress, exercise_freq=exercise,
                 mood=mood, wellness_score=score, score_category=cat)
    db.session.add(r); db.session.commit()
    flash(f'Score: {score}/100 — {cat.replace("-"," ").title()} ✨', 'success')
    return redirect(url_for('dashboard'))

@app.route('/mantra')
@login_required
def mantra_page():
    return render_template('mantra_suggest.html')

@app.route('/player')
@login_required
def player():
    return render_template('mantra_player.html')

@app.route('/history')
@login_required
def history():
    user = User.query.get(session['user_id'])
    all_r = Response.query.filter_by(user_id=user.id).order_by(Response.timestamp.desc()).all()
    for r in all_r:
        r.struggles_list = json.loads(r.struggles) if r.struggles else []
    return render_template('history.html', user=user, responses=all_r)

# ── DL Prediction API (with ML Fallback) ────────────────────────
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if not dl_model and not ml_model:
        return jsonify({'success':False, 'error':'No model loaded. Run train_dl_enhanced_model.py first.'})
    try:
        d = request.get_json()
        # Map frontend mantra IDs to encoder class names
        MANTRA_MAP = {'om':'Om','gayatri':'Gayatri','om_mani':'Om_Mani',
                      'shma':'Shma','sufi':'Sufi_Dhikr','hesychasm':'Hesychasm'}
        GENDER_MAP = {'male':'M','female':'F','other':'M'}

        raw_mantra = d['mantra']
        mantra = MANTRA_MAP.get(raw_mantra, raw_mantra)
        if mantra not in le_mantra.classes_:
            mantra = le_mantra.classes_[0]

        raw_gender = d['gender']
        gender = GENDER_MAP.get(raw_gender, raw_gender)
        if gender not in le_gender.classes_:
            gender = le_gender.classes_[0]

        duration = int(d['duration'])
        rpm = int(d['rpm']); breath_sync = int(d['breath_sync'])
        pre_stress = int(d['pre_stress']); pre_anxiety = int(d['pre_anxiety'])
        pre_focus  = int(d['pre_focus']);  pre_calm    = int(d['pre_calm'])
        age = int(d['age']); experience = d['experience']

        ef = {'beginner':0.75,'intermediate':1.0,'advanced':1.25}[experience]
        df_val = min(duration/15.0, 2.0)
        cort   = round(-20.0 * df_val * ef, 1)
        hrv    = round( 12.0 * df_val * ef, 1)
        alpha  = round( 24.0 * df_val * ef, 1)
        theta  = round( 20.0 * df_val * ef, 1)
        sr = pre_stress  * abs(cort)/100 * 0.9
        ar = pre_anxiety * abs(cort)/100 * 0.85
        fg = (100-pre_focus) * alpha/100 * 0.7
        cg = (100-pre_calm)  * alpha/100 * 0.75
        total = sr+ar+fg+cg

        sample = {
            'mantra_enc': int(le_mantra.transform([mantra])[0]),
            'duration_minutes': duration, 'repetitions_per_min': rpm,
            'breath_sync_sec': breath_sync, 'pre_stress': pre_stress,
            'pre_anxiety': pre_anxiety, 'pre_focus': pre_focus,
            'pre_calm': pre_calm, 'hrv_change': hrv,
            'cortisol_change_percent': cort, 'alpha_wave_increase': alpha,
            'theta_wave_increase': theta, 'age': age,
            'gender_enc': int(le_gender.transform([gender])[0]),
            'exp_enc': int(le_exp.transform([experience])[0]),
            'stress_reduction': round(sr,1), 'anxiety_reduction': round(ar,1),
            'focus_gain': round(fg,1), 'calm_gain': round(cg,1),
            'total_improvement': round(total,1),
            'wellness_score': round(total+hrv,1),
            'stress_anxiety_ratio': round(sr/(ar+0.01),2),
            'biofeedback_composite': round((hrv + abs(cort) + alpha + theta)/4, 1),
            'session_intensity': round(df_val * rpm * ef, 1)
        }
        sdf = pd.DataFrame([sample])[FEATURE_COLS]
        sdf_scaled = dl_scaler.transform(sdf)

        # ── Try DL model first, fallback to ML ──────────────────
        model_used = None; model_used_key = None; dl_failed = False
        pred = None; pred_prob = None

        if dl_model:
            try:
                if MODEL_TYPE == 'tensorflow':
                    pred_prob = dl_model.predict(sdf_scaled, verbose=0)[0]
                    pred = np.argmax(pred_prob)
                else:
                    pred = dl_model.predict(sdf_scaled)[0]
                    pred_prob = dl_model.predict_proba(sdf_scaled)[0] if hasattr(dl_model, 'predict_proba') else np.eye(len(le_target.classes_))[int(pred)]
                model_used = 'DL'; model_used_key = 'DL_MLP'
            except Exception as dl_err:
                print(f"DL model failed: {dl_err}, falling back to ML")
                dl_failed = True

        if model_used is None or dl_failed:
            if ml_model:
                pred = ml_model.predict(sdf_scaled)[0]
                pred_prob = ml_model.predict_proba(sdf_scaled)[0] if hasattr(ml_model, 'predict_proba') else np.eye(len(le_target.classes_))[int(pred)]
                model_used = 'ML'; model_used_key = 'ML_GB'
            else:
                return jsonify({'success':False,'error':'Both DL and ML models failed.'})

        eff  = le_target.inverse_transform([pred])[0]
        probs = {c:round(float(p),3) for c,p in zip(le_target.classes_, pred_prob)}
        model_info = MODEL_COMPARISON.get(model_used_key, {})

        return jsonify({
            'success': True, 'effectiveness': eff, 'probabilities': probs,
            'model_used': model_used, 'model_used_key': model_used_key,
            'dl_failed': dl_failed,
            'model_name': model_info.get('name', model_used),
            'model_color': model_info.get('color', '#F4C542'),
            'model_icon': model_info.get('icon', '🤖'),
            'model_accuracy': model_info.get('accuracy'),
            'model_f1': model_info.get('f1_score'),
            'comparison': MODEL_COMPARISON,
            'metrics': {
                'post_stress':round(max(10,pre_stress-sr),1),
                'post_anxiety':round(max(8,pre_anxiety-ar),1),
                'post_focus':round(min(95,pre_focus+fg),1),
                'post_calm':round(min(98,pre_calm+cg),1),
                'stress_reduction':round(sr,1),'anxiety_reduction':round(ar,1),
                'focus_gain':round(fg,1),'calm_gain':round(cg,1),
                'hrv_change':hrv,'cortisol_change':cort,
                'alpha_increase':alpha,'theta_increase':theta
            }
        })
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 400


@app.route('/api/model_comparison')
def model_comparison_api():
    return jsonify(MODEL_COMPARISON)

@app.route('/api/affirmation')
def api_affirmation():
    return jsonify({'affirmation': random.choice(AFFIRMATIONS)})

@app.route('/api/stats')
def api_stats():
    try:
        df = pd.read_csv(os.path.join(BASE_DIR,'dataset','mantra_dataset.csv'))
        return jsonify({
            'total_sessions': len(df),
            'avg_stress_reduction': round((df['pre_stress']-df['post_stress']).mean(),1),
            'avg_alpha_increase': round(df['alpha_wave_increase'].mean(),1),
            'best_mantra': df.groupby('mantra_type')['cortisol_change_percent'].mean().abs().idxmax()
        })
    except: return jsonify({'total_sessions':200})

with app.app_context():
    os.makedirs('instance', exist_ok=True)
    db.create_all()
    print("Database ready")

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Mantra & Mind Web App")
    print("  http://localhost:5000")
    print("="*50+"\n")
    app.run(debug=True, port=5000)
