import os
import pickle
import pandas as pd
import numpy as np
import sqlite3
from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# 1. Model Loading
# -----------------------------
MODEL_PATH = "investment_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

def load_pickle(path):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    return None

# Load only once when server starts
model = load_pickle(MODEL_PATH)
label_encoder = load_pickle(ENCODER_PATH)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'super_secret_investment_key'

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_inv_db_connection():
    conn = sqlite3.connect('investments.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_investments_db():
    conn = get_inv_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS investment_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            amount INTEGER,
            risk_level TEXT,
            duration INTEGER,
            recommendation TEXT,
            date TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_investments_db()

# -----------------------------
# 2. Flask Routes
# -----------------------------
@app.route("/")
def splash():
    return render_template("splash.html")

@app.route("/home")
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    print("Homepage loaded")
    username = session.get('username', 'Investor')
    return render_template("index.html", username=username)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('home'))  # redirect to /home after login
        else:
            return render_template("login.html", error="Invalid username or password")
            
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        hashed_pw = generate_password_hash(password)
        
        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, hashed_pw))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Username or email already exists")
            
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

# -----------------------------
# 3. Prediction & Validation Logic
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    print("Prediction request received")
    
    if not model:
        return render_template("error.html", message="investment_model.pkl not found. Please train the model first.")

    try:
        # Get values from the form
        age_raw = request.form.get("age")
        monthly_income_raw = request.form.get("monthly_income")
        savings_raw = request.form.get("savings")
        duration_months_raw = request.form.get("duration_months")
        investment_amount_raw = request.form.get("investment_amount")
        frequency = request.form.get("frequency")
        risk_level = request.form.get("risk_level")
        investment_mode = request.form.get("investment_mode")

        # -----------------------------
        # 4. Validation Rules
        # -----------------------------
        if not all([age_raw, monthly_income_raw, savings_raw, duration_months_raw, investment_amount_raw, frequency]):
            return render_template("error.html", message="All numeric fields are required.")

        try:
            age = float(age_raw)
            monthly_income = float(monthly_income_raw)
            savings = float(savings_raw)
            duration_months = int(duration_months_raw)
            investment_amount = float(investment_amount_raw)
        except ValueError:
            return render_template("error.html", message="Numeric fields must contain valid numbers.")

        if monthly_income < savings:
            return render_template("error.html", message="Monthly Income cannot be less than savings.")
            
        if investment_amount > savings:
            return render_template("error.html", message="Investment amount cannot exceed savings.")
            
        if duration_months < 1 or duration_months > 120:
            return render_template("error.html", message="Please enter an investment duration between 1 and 120 months.")
            
        if investment_amount < 100:
            return render_template("error.html", message="Minimum SIP investment should be ₹100.")

        if frequency == "daily":
            monthly_equivalent = investment_amount * 30
        elif frequency == "weekly":
            monthly_equivalent = investment_amount * 4
        elif frequency == "monthly":
            monthly_equivalent = investment_amount
        else:
            monthly_equivalent = investment_amount

        market_volatility_index = 20.0
        interest_rate = 4.0
        inflation_rate = 3.0
        stock_index_change = 1.0

        # Prepare the input dataframe for the model
        input_data = pd.DataFrame([{
            "age": age,
            "monthly_income": monthly_income,
            "savings": savings,
            "investment_amount": monthly_equivalent,
            "duration_months": duration_months,
            "risk_level": risk_level
        }])

        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_encoded.columns
        for col in expected_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[expected_columns]

        # Predict the recommendation
        prediction_numeric = model.predict(input_encoded)[0]
        
        # Decode the result
        if label_encoder:
            rounded_pred = int(round(prediction_numeric))
            max_class_idx = len(label_encoder.classes_) - 1
            rounded_pred = max(0, min(rounded_pred, max_class_idx))
            prediction_text = label_encoder.inverse_transform([rounded_pred])[0]
        else:
            prediction_text = str(prediction_numeric)

        # -----------------------------
        # SIP Growth Calculator (12% annual return)
        # -----------------------------
        sip_p = monthly_equivalent
        sip_r = 0.12 / 12  # monthly rate
        sip_n = duration_months
        sip_fv = sip_p * (((1 + sip_r) ** sip_n - 1) * (1 + sip_r)) / sip_r
        sip_invested = sip_p * sip_n
        sip_profit = sip_fv - sip_invested
        sip_fv = round(sip_fv, 2)
        sip_invested = round(sip_invested, 2)
        sip_profit = round(sip_profit, 2)

        # -----------------------------
        # Save to investment_history
        # -----------------------------
        from datetime import datetime
        try:
            username = session.get('username', 'guest')
            inv_conn = get_inv_db_connection()
            inv_conn.execute(
                'INSERT INTO investment_history (username, amount, risk_level, duration, recommendation, date) VALUES (?, ?, ?, ?, ?, ?)',
                (username, int(monthly_equivalent), risk_level, duration_months, prediction_text, datetime.now().strftime('%Y-%m-%d %H:%M'))
            )
            inv_conn.commit()
            inv_conn.close()
        except Exception:
            pass


        # Build risk meter details (used by both single and diversified)
        if risk_level == "Low":
            risk_meter = "██░░░░░░"
            risk_label = "Conservative Investor"
        elif risk_level == "Medium":
            risk_meter = "████░░░░"
            risk_label = "Balanced Investor"
        else:
            risk_meter = "████████"
            risk_label = "Aggressive Investor"

        # Build risk-based platform lists (lists of dictionaries)
        if risk_level == "Low":
            platforms = [
                {"name": "Groww",  "desc": "Beginner-friendly platform for mutual funds and SIP investing.",        "url": "https://groww.in"},
                {"name": "Upstox", "desc": "Simple and fast platform for stocks, ETFs, and basic investing.",       "url": "https://upstox.com"}
            ]
        elif risk_level == "Medium":
            platforms = [
                {"name": "Groww",   "desc": "Invest in mutual funds, ETFs, and stocks on one platform.",            "url": "https://groww.in"},
                {"name": "Zerodha", "desc": "India's leading platform with advanced charting and direct funds.",    "url": "https://zerodha.com"}
            ]
        elif risk_level == "High":
            platforms = [
                {"name": "Zerodha",   "desc": "India's leading advanced platform for stock and ETF trading.",       "url": "https://zerodha.com"},
                {"name": "Angel One", "desc": "Research-driven investment platform with advanced trading tools.",   "url": "https://www.angelone.in"}
            ]
        else:
            platforms = [
                {"name": "Groww",  "desc": "Beginner-friendly platform.", "url": "https://groww.in"},
                {"name": "Upstox", "desc": "Simple investing platform.",  "url": "https://upstox.com"}
            ]

        if investment_mode == "single":
            return render_template(
                "result.html",
                prediction=prediction_text,
                frequency=frequency.capitalize(),
                investment_amount=investment_amount,
                monthly_equivalent=monthly_equivalent,
                sip_invested=sip_invested,
                sip_fv=sip_fv,
                sip_profit=sip_profit,
                risk_level=risk_level,
                risk_meter=risk_meter,
                risk_label=risk_label,
                platforms=platforms
            )

        # Map risk level and duration to portfolio allocation
        base_allocation = {}
        if duration_months <= 24: # Short Term
            if risk_level == "Low":
                base_allocation = {"Fixed Deposit": 60, "Bonds": 40}
            elif risk_level == "Medium":
                base_allocation = {"Bonds": 50, "Gold": 30, "Fixed Deposit": 20}
            else: # High
                base_allocation = {"Gold": 60, "Bonds": 40}
        elif duration_months <= 60: # Medium Term
            if risk_level == "Low":
                base_allocation = {"Bonds": 50, "Fixed Deposit": 30, "Index Funds": 20}
            elif risk_level == "Medium":
                base_allocation = {"Mutual Funds": 40, "Gold": 30, "Bonds": 30}
            else: # High
                base_allocation = {"Index Funds": 40, "Mutual Funds": 30, "Gold": 30}
        else: # Long Term
            if risk_level == "Low":
                base_allocation = {"Index Funds": 50, "Mutual Funds": 30, "Bonds": 20}
            elif risk_level == "Medium":
                base_allocation = {"ETFs": 40, "Mutual Funds": 40, "Index Funds": 20}
            else: # High
                base_allocation = {"Stocks": 50, "SIP Mutual Funds": 30, "Crypto": 20}

        portfolio = {}
        for asset, percentage in base_allocation.items():
            allocated_amount = int(investment_amount * percentage / 100)
            portfolio[asset] = {"percent": percentage, "amount": allocated_amount}


        return render_template(
            "result.html",
            portfolio=portfolio,
            risk_level=risk_level,
            risk_meter=risk_meter,
            risk_label=risk_label,
            frequency=frequency.capitalize(),
            investment_amount=investment_amount,
            monthly_equivalent=monthly_equivalent,
            sip_invested=sip_invested,
            sip_fv=sip_fv,
            sip_profit=sip_profit,
            platforms=platforms
        )

    except Exception as e:
        return render_template("error.html", message=f"An error occurred during prediction: {str(e)}")

# -----------------------------
# 5. Chatbot Logic
# -----------------------------
@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    response_text = ""
    question_original = ""
    
    try:
        if request.method == "POST":
            question_original = request.form.get("question", "")
            if question_original:
                question = question_original.lower()
                
                # Chatbot Intent Detection
                if any(phrase in question for phrase in ["who are you", "who r u", "what are you", "introduce yourself", "your name", "what is your name"]):
                    response_text = "I am your Smart Financial Advisor from InvestSmart. I help you understand investments, financial planning, and guide you in making smarter financial decisions."
                elif any(word in question for word in ["risk", "risk level", "risk tolerance"]):
                    response_text = "Choosing a risk level depends on your financial goals. Low risk investments include bonds or fixed deposits. Medium risk includes mutual funds or ETFs. High risk includes stocks or cryptocurrency."
                elif any(word in question for word in ["safe", "trust", "secure", "fear", "scared", "afraid", "nervous", "worried"]):
                    response_text = "Investing always involves some risk, but you can reduce it by diversifying your investments, investing for the long term, and avoiding putting all your money into one asset."
                elif any(phrase in question for phrase in ["how should i start", "don’t know anything", "beginner", "start investing", "first time", "start", "begin", "new investor"]):
                    response_text = "If you are a beginner, start by building an emergency fund and then consider investing small amounts in diversified options such as mutual funds or index funds."
                elif any(phrase in question for phrase in ["small amount", "1000", "2000", "5000"]):
                    response_text = "If you have a small amount like 2000, starting a SIP in a mutual fund is a good option. SIP allows you to invest small amounts regularly and gradually build wealth."
                elif any(phrase in question for phrase in ["sip", "systematic investment plan", "start sip", "sip investment", "sip mutual fund", "how does sip work"]):
                    response_text = "SIP stands for Systematic Investment Plan. It allows investors to invest a fixed amount of money regularly in mutual funds. SIP is popular among beginners because it allows small investments and reduces market timing risk through rupee cost averaging."
                elif any(phrase in question for phrase in ["is sip good", "should i start sip", "is sip safe", "how much should i invest in sip"]):
                    response_text = "SIP is one of the best investment methods for beginners because it allows small regular investments instead of a large one-time investment. Over time, SIP benefits from compounding and disciplined investing."
                elif any(word in question for word in ["mutual", "mutual fund", "mutual funds"]):
                    response_text = "Mutual funds pool money from many investors and invest in diversified financial assets like stocks and bonds."
                elif any(word in question for word in ["stock", "share market", "shares", "stocks"]):
                    response_text = "Stocks represent ownership in a company and offer high growth potential but also involve higher risk."
                elif any(word in question for word in ["diversification", "diversify", "allocate", "portfolio diversification"]):
                    response_text = "Diversification means spreading investments across different asset types to reduce overall investment risk."
                elif any(word in question for word in ["emergency", "emergency fund", "saving money"]):
                    response_text = "An emergency fund is savings set aside to cover unexpected expenses before starting investments."
                elif any(word in question for word in ["etf", "etfs", "exchange traded fund"]):
                    response_text = "An ETF (Exchange Traded Fund) is a type of investment fund that trades on the stock exchange like a stock. ETFs usually track an index such as Nifty 50 or S&P 500 and allow investors to invest in multiple companies through a single investment."
                elif any(word in question for word in ["bond", "bonds"]):
                    response_text = "Bonds are fixed income investments that provide stable but generally lower returns compared to stocks."
                elif "gold" in question:
                    response_text = "Gold is considered a safe investment that can help protect wealth during inflation or economic uncertainty."
                elif any(word in question for word in ["crypto", "bitcoin", "cryptocurrency", "ethereum"]):
                    response_text = "Cryptocurrencies are high-risk digital assets that can offer high returns but also involve significant volatility."
                elif any(word in question for word in ["long term", "long-term", "investing basics"]):
                    response_text = "Long-term investing allows your wealth to grow over time through compound interest and helps you ride out market volatility."

            if not response_text:
                response_text = "That's an interesting financial question. A good starting point for beginners is building an emergency fund and investing regularly through SIP in diversified mutual funds."

            return render_template("chatbot.html", question=question_original, response=response_text)
            
        return render_template("chatbot.html", question=question_original, response=None)
    except Exception as e:
        return render_template("error.html", message=f"Chatbot encountered an error: {str(e)}")

# -----------------------------
# 6. Investment History Dashboard
# -----------------------------
@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    username = session.get('username', 'guest')
    conn = get_inv_db_connection()
    history = conn.execute(
        'SELECT * FROM investment_history WHERE username = ? ORDER BY id DESC',
        (username,)
    ).fetchall()
    conn.close()
    return render_template("dashboard.html", history=history, username=username)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
     