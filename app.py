from flask import Flask, request, render_template, redirect, url_for, session, jsonify
import joblib
import os
import json

app = Flask(__name__)
app.secret_key = 'vku_mlops_secret_key'

# --- 1. CONFIG DATA ---
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Định nghĩa Khối xét tuyển và Môn thi
BLOCKS = {
    "A00": ["Toán", "Vật lí", "Hóa học"],
    "A01": ["Toán", "Vật lí", "Tiếng Anh"],
    "A03": ["Toán", "Vật lí", "Lịch sử"],
    "B00": ["Toán", "Hóa học", "Sinh học"],
    "B03": ["Toán", "Sinh học", "Ngữ văn"],
    "B08": ["Toán", "Sinh học", "Tiếng Anh"],
    "C00": ["Ngữ văn", "Lịch sử", "Địa lí"],
    "C20": ["Ngữ văn", "Địa lí", "GDKTPL"],
    "C04": ["Ngữ văn", "Toán", "Địa lí"],
    "D01": ["Ngữ văn", "Toán", "Tiếng Anh"],
    "D07": ["Toán", "Hóa học", "Tiếng Anh"],
    "D84": ["Toán", "Tiếng Anh", "GDKTPL"]
}

HISTORICAL_DATA = {
    0: { "name": "Công nghệ thông tin (CNTT)", "scores": [23.0, 25.0, 25.01, 25.0, 20.0] },
    1: { "name": "Trí tuệ nhân tạo (AI)", "scores": [21.05, 25.0, 25.01, 25.0, 21.0] },
    2: { "name": "An toàn thông tin", "scores": [22.0, 24.5, 23.0, 24.0, 19.0] },
    3: { "name": "Công nghệ kỹ thuật máy tính / Vi mạch", "scores": [20.0, 24.0, 23.0, 27.0, 24.0] },
    4: { "name": "Marketing", "scores": [22.0, 25.0, 23.0, 26.0, 23.25] },
    5: { "name": "Logistics & Chuỗi cung ứng số", "scores": [23.0, 25.0, 23.0, 26.0, 23.5] },
    6: { "name": "Quản trị kinh doanh (QTKD)", "scores": [22.5, 24.0, 23.0, 25.0, 22.0] },
    7: { "name": "Du lịch & Lữ hành số", "scores": [20.5, 24.0, 22.5, 25.0, 23.0] },
    8: { "name": "Công nghệ tài chính", "scores": [21.0, 24.0, 22.5, 24.0, 22.0] }
}
YEARS = [2021, 2022, 2023, 2024, 2025]
MAJORS = {k: v["name"] for k, v in HISTORICAL_DATA.items()}

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html', page='home', majors=MAJORS, blocks=BLOCKS)

# --- NEW: API HEALTH CHECK ---
@app.route('/health', methods=['GET'])
def health_check():
    """
    API kiểm tra trạng thái hệ thống.
    Trả về JSON: { "status": "UP", "model_loaded": boolean }
    """
    try:
        # Kiểm tra xem model có đang hoạt động không
        is_model_ok = model is not None
        
        status_info = {
            "status": "UP",
            "service": "VKU AI Prediction System",
            "model_loaded": is_model_ok
        }
        # Trả về code 200 OK
        return jsonify(status_info), 200
    except Exception as e:
        # Nếu server lỗi, trả về code 500
        return jsonify({"status": "DOWN", "error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process():
    try:
        major_id = int(request.form['major_id'])
        block_id = request.form['block_id']
        s1 = float(request.form['score1'])
        s2 = float(request.form['score2'])
        s3 = float(request.form['score3'])
        tong_diem = s1 + s2 + s3
        
        du_doan = 0
        if model:
            du_doan = model.predict([[2026, major_id]])[0]
            
        session['result'] = {
            'major_id': major_id,
            'block_id': block_id,
            'tong_diem': round(tong_diem, 2),
            'diem_chuan': round(du_doan, 2)
        }
        return redirect(url_for('result'))
    except Exception as e:
        return f"Lỗi nhập liệu: {str(e)}"

@app.route('/result')
def result():
    data = session.get('result')
    if not data or 'block_id' not in data:
        return redirect(url_for('index'))
    
    chenh_lech = data['tong_diem'] - data['diem_chuan']
    
    if chenh_lech >= 2.0:
        ty_le, msg, color = 99, "CHÚC MỪNG! BẠN RẤT AN TOÀN", "#059669"
    elif chenh_lech >= 0.5:
        ty_le, msg, color = 90, "CƠ HỘI TRÚNG TUYỂN CAO", "#10B981"
    elif chenh_lech >= -0.5:
        ty_le, msg, color = 60, "CẠNH TRANH GAY GẮT", "#F59E0B"
    elif chenh_lech >= -2.0:
        ty_le, msg, color = 30, "CẦN CỐ GẮNG NHIỀU HƠN", "#EF4444"
    else:
        ty_le, msg, color = 10, "HÃY CHỌN NGUYỆN VỌNG KHÁC", "#991B1B"
        
    return render_template('result.html', page='result', has_data=True, 
                           ten_nganh=MAJORS.get(data['major_id']),
                           khoi_xet=data['block_id'],
                           tong_diem=data['tong_diem'],
                           diem_chuan=data['diem_chuan'],
                           ty_le=int(ty_le),
                           loi_khuyen=msg,
                           color_code=color)

@app.route('/analytics')
def analytics():
    chart_data = {
        "labels": YEARS,
        "datasets": []
    }
    colors = ['#4F46E5', '#EC4899', '#F59E0B', '#10B981', '#3B82F6', '#6366F1', '#8B5CF6', '#EC4899', '#14B8A6']
    
    for idx, (mid, mdata) in enumerate(HISTORICAL_DATA.items()):
        chart_data["datasets"].append({
            "label": mdata["name"],
            "data": mdata["scores"],
            "borderColor": colors[idx % len(colors)],
            "fill": False,
            "tension": 0.4
        })

    return render_template('analytics.html', page='analytics', 
                           chart_json=json.dumps(chart_data))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
