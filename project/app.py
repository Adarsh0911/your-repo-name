from flask import Flask, render_template, request, redirect, session, url_for, flash
import sqlite3
import pickle
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load ML model and label encoder
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize DB
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT)''')
    conn.close()

init_db()

# Home/Login Page
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']

        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=?", (uname,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[2], pwd):
            session['username'] = uname
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials", "danger")
    return render_template("index.html")

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = generate_password_hash(request.form['password'])

        try:
            conn = sqlite3.connect('users.db')
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (uname, pwd))
            conn.commit()
            conn.close()
            flash("Registration successful!", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists", "warning")

    return render_template("register.html")
 
#predict
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect('/')
    
    if request.method == 'POST':
        try:
            values = [float(request.form[k]) for k in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
            predicted_crop = model.predict([values])
            crop = encoder.inverse_transform(predicted_crop)[0].lower()
            bg_image = f"{crop}.jpg"  # Make sure you have an image with this name

            return render_template('result.html', crop=crop, bg_image=bg_image)
        except Exception as e:
            return render_template('result.html', crop=f"Error: {e}", bg_image="bg.jpg")
    
    return render_template('predict.html')



# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)
model = pickle.load(open("crop_model.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))


