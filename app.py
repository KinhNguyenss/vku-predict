from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import os
import json
import numpy as np

app = Flask(__name__)
app.secret_key = 'vku_mlops_secret_key'

# --- CONFIG & DATA ---
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

HISTORICAL_DATA = {
    0: { "name": "Công nghệ thông tin", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [17.0, 19.75, 18.0, 23.0, 25.0, 23.5, 24.0, 24.71] },
    1: { "name": "Logistics & Chuỗi cung ứng", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [16.0, 18.0, 18.0, 23.0, 25.0, 23.0, 26.0, 26.3] },
    2: { "name": "Quản trị kinh doanh", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [16.0, 18.75, 18.0, 22.5, 24.0, 23.0, 25.0, 25.65] }
}
MAJORS = {k: v["name"] for k, v in HISTORICAL_DATA.items()}

# --- ROUTES ---

@app.route('/')
def index():
    # Sử dụng render_template thay vì render_template_string
    return render_template('index.html', page='home', majors=MAJORS)

@app.route('/process', methods=['POST'])
def process():
    try:
        major_id = int(request.form['major_id'])
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        tong_diem = toan + ly + hoa
        
        du_doan = 0
        if model:
            # Model LinearRegression yêu cầu input 2 chiều
            du_doan = model.predict([[2026, major_id]])[0]
            
        session['result'] = {
            'major_id': major_id,
            'tong_diem': round(tong_diem, 2),
            'diem_chuan': round(du_doan, 2)
        }
        return redirect(url_for('result'))
    except Exception as e:
        return f"Lỗi: {str(e)}"

@app.route('/result')
def result():
    data = session.get('result')
    if not data:
        return render_template('result.html', page='result', has_data=False)
    
    chenh_lech = data['tong_diem'] - data['diem_chuan']
    if chenh_lech >= 1.5:
        ty_le, msg, color = 99, "CỰC KỲ AN TOÀN", "#059669"
    elif chenh_lech >= 0:
        ty_le, msg, color = 85 + int(chenh_lech*5), "KHẢ QUAN", "#10B981"
    elif chenh_lech >= -1.0:
        ty_le, msg, color = 50 + int(chenh_lech*30), "CẦN CÂN NHẮC", "#F59E0B"
    else:
        ty_le, msg, color = max(5, 50 + int(chenh_lech*20)), "RỦI RO CAO", "#EF4444"
        
    return render_template('result.html', page='result', has_data=True, 
                           ten_nganh=MAJORS.get(data['major_id']),
                           tong_diem=data['tong_diem'],
                           diem_chuan=data['diem_chuan'],
                           ty_le=int(ty_le),
                           loi_khuyen=msg,
                           color_code=color)

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', page='analytics', 
                           data=HISTORICAL_DATA, 
                           chart_json=json.dumps(HISTORICAL_DATA))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
