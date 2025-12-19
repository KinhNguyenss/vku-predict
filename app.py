from flask import Flask, request, render_template_string, redirect, url_for, session
import joblib
import os
import json
import numpy as np

app = Flask(__name__)
app.secret_key = 'vku_mlops_secret_key' # Cần key để lưu dữ liệu tạm thời giữa các trang

# --- 1. CONFIG & DATA ---
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Dữ liệu biểu đồ (Analytics)
HISTORICAL_DATA = {
    0: { "name": "Công nghệ thông tin", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [17.0, 19.75, 18.0, 23.0, 25.0, 23.5, 24.0, 24.71] },
    1: { "name": "Logistics & Chuỗi cung ứng", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [16.0, 18.0, 18.0, 23.0, 25.0, 23.0, 26.0, 26.3] },
    2: { "name": "Quản trị kinh doanh", "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], "scores": [16.0, 18.75, 18.0, 22.5, 24.0, 23.0, 25.0, 25.65] }
}
MAJORS = {k: v["name"] for k, v in HISTORICAL_DATA.items()}

# --- 2. HTML TEMPLATES (CHIA TÁCH THÀNH CÁC BIẾN RIÊNG) ---

# Phần chung (Header, Style)
BASE_HEAD = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VKU AI Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #F8F9FE; color: #1F2937; }
        .navbar { background: white; box-shadow: 0 1px 2px rgba(0,0,0,0.05); padding: 1rem 2rem; }
        .navbar-brand { font-weight: 700; color: #4F46E5; font-size: 1.4rem; }
        .nav-link { color: #6B7280; font-weight: 500; margin: 0 10px; transition: 0.3s; }
        .nav-link.active { color: #4F46E5; font-weight: 700; background: #EEF2FF; border-radius: 8px; }
        .nav-link:hover { color: #4F46E5; }
        
        .main-card { background: white; border-radius: 16px; border: 1px solid #E5E7EB; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); padding: 30px; margin-top: 20px; min-height: 550px; }
        .btn-primary-custom { background: #4F46E5; color: white; border: none; padding: 12px; border-radius: 8px; width: 100%; font-weight: 600; }
        .btn-primary-custom:hover { background: #4338CA; }
        
        .history-item { background: white; padding: 12px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #E5E7EB; font-size: 0.9rem; }
        .history-sidebar { background: #F9FAFB; border-left: 1px solid #E5E7EB; padding: 20px; height: 100%; border-radius: 0 16px 16px 0; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg sticky-top">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-brain me-2"></i>AI Prediction Platform</a>
            <div class="collapse navbar-collapse justify-content-end">
                <div class="navbar-nav">
                    <a class="nav-link {active_predict}" href="/">New Prediction</a>
                    <a class="nav-link {active_result}" href="/result">Results</a>
                    <a class="nav-link {active_analytics}" href="/analytics">Analytics</a>
                </div>
            </div>
        </div>
    </nav>
"""

# PAGE 1: NEW PREDICTION (Form + History Sidebar)
PAGE_PREDICT = BASE_HEAD + """
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-card p-0 overflow-hidden">
                    <div class="row g-0 h-100">
                        <div class="col-md-8 p-4 p-md-5">
                            <h4 class="mb-2 fw-bold text-dark">AI Prediction Engine</h4>
                            <p class="text-muted mb-4 small">Nhập thông tin điểm số để AI phân tích khả năng trúng tuyển.</p>
                            
                            <form method="post" action="/process">
                                <div class="mb-4">
                                    <label class="form-label fw-bold">Chọn Chuyên Ngành</label>
                                    <select name="major_id" class="form-select bg-light border-0 py-3">
                                        {% for id, name in majors.items() %}
                                            <option value="{{ id }}">{{ name }}</option>
                                        {% endfor %}
                                    </select>
                                </div>

                                <div class="row g-3 mb-4">
                                    <div class="col-md-4">
                                        <label class="form-label small text-muted">TOÁN</label>
                                        <input type="number" step="0.1" name="toan" class="form-control bg-light border-0 py-3" placeholder="0.0" required>
                                    </div>
                                    <div class="col-md-4">
                                        <label class="form-label small text-muted">LÝ</label>
                                        <input type="number" step="0.1" name="ly" class="form-control bg-light border-0 py-3" placeholder="0.0" required>
                                    </div>
                                    <div class="col-md-4">
                                        <label class="form-label small text-muted">HÓA/ANH</label>
                                        <input type="number" step="0.1" name="hoa" class="form-control bg-light border-0 py-3" placeholder="0.0" required>
                                    </div>
                                </div>

                                <button type="submit" class="btn btn-primary-custom mt-2">
                                    <i class="fas fa-magic me-2"></i>Generate Prediction
                                </button>
                            </form>
                        </div>

                        <div class="col-md-4 history-sidebar">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h6 class="fw-bold mb-0"><i class="fas fa-history me-2"></i>Lịch sử</h6>
                                <button onclick="clearHistory()" class="btn btn-sm text-danger p-0" style="font-size: 0.8rem;">Xóa</button>
                            </div>
                            <div id="history-list" style="max-height: 450px; overflow-y: auto;">
                                <div class="text-center text-muted mt-5 small">Chưa có dữ liệu</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function renderHistory() {
            let history = JSON.parse(localStorage.getItem('vku_history')) || [];
            const container = document.getElementById('history-list');
            if (history.length === 0) {
                container.innerHTML = '<div class="text-center text-muted mt-5 small">Chưa có dữ liệu</div>';
                return;
            }
            let html = '';
            history.forEach(item => {
                html += `
                    <div class="history-item shadow-sm">
                        <div class="fw-bold text-primary" style="font-size: 0.8rem">${item.nganh}</div>
                        <div class="d-flex justify-content-between mt-1">
                            <span class="small text-muted">Điểm: ${item.diem}</span>
                            <span class="badge bg-${item.color}">${item.tyle}%</span>
                        </div>
                        <div class="text-end text-muted" style="font-size: 0.65rem; margin-top:4px;">${item.time}</div>
                    </div>`;
            });
            container.innerHTML = html;
        }
        function clearHistory() { localStorage.removeItem('vku_history'); renderHistory(); }
        renderHistory();
    </script>
</body></html>
"""

# PAGE 2: RESULTS (Show Result)
PAGE_RESULT = BASE_HEAD + """
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="main-card text-center d-flex flex-column justify-content-center align-items-center">
                    {% if has_data %}
                        <div class="mb-4">
                            <div class="badge bg-light text-primary px-3 py-2 rounded-pill mb-3">Kết quả dự đoán 2026</div>
                            <h2 class="fw-bold mb-1">{{ ten_nganh }}</h2>
                            <p class="text-muted">Dựa trên dữ liệu học máy từ 2018-2025</p>
                        </div>

                        <div class="row w-100 justify-content-center mb-5">
                            <div class="col-md-5 mb-3">
                                <div class="p-4 border rounded-4 bg-light">
                                    <small class="text-muted text-uppercase fw-bold">Điểm của bạn</small>
                                    <div class="display-4 fw-bold text-dark mt-2">{{ tong_diem }}</div>
                                </div>
                            </div>
                            <div class="col-md-5 mb-3">
                                <div class="p-4 border rounded-4 bg-light">
                                    <small class="text-muted text-uppercase fw-bold">Điểm chuẩn AI</small>
                                    <div class="display-4 fw-bold text-primary mt-2">{{ diem_chuan }}</div>
                                </div>
                            </div>
                        </div>

                        <div class="mb-4">
                            <h5 class="text-muted mb-3">Tỷ lệ trúng tuyển</h5>
                            <div class="display-1 fw-bold" style="color: {{ color_code }}">{{ ty_le }}%</div>
                            <p class="mt-2 fs-5 badge px-4 py-2" style="background-color: {{ color_code }}; color: white;">{{ loi_khuyen }}</p>
                        </div>

                        <a href="/" class="btn btn-outline-dark px-5 py-2 rounded-pill">
                            <i class="fas fa-arrow-left me-2"></i>Dự đoán khác
                        </a>

                        <script>
                            window.onload = function() {
                                let history = JSON.parse(localStorage.getItem('vku_history')) || [];
                                const newRecord = {
                                    nganh: "{{ ten_nganh }}",
                                    diem: "{{ tong_diem }}",
                                    tyle: "{{ ty_le }}",
                                    color: "{{ 'success' if ty_le > 80 else 'warning' if ty_le > 50 else 'danger' }}",
                                    time: new Date().toLocaleTimeString()
                                };
                                // Chỉ lưu nếu bản ghi mới nhất không trùng (tránh F5 lưu trùng)
                                if(history.length === 0 || history[0].time !== newRecord.time) {
                                     // Lưu ý: demo đơn giản, thực tế nên check ID request
                                     history.unshift(newRecord);
                                     if(history.length > 10) history.pop();
                                     localStorage.setItem('vku_history', JSON.stringify(history));
                                }
                            };
                        </script>
                    {% else %}
                        <div class="text-muted">
                            <i class="fas fa-ghost fa-3x mb-3"></i>
                            <p>Chưa có dữ liệu. Vui lòng nhập thông tin.</p>
                            <a href="/" class="btn btn-primary-custom w-auto px-4">Đến trang nhập liệu</a>
                        </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body></html>
"""

# PAGE 3: ANALYTICS (Charts)
PAGE_ANALYTICS = BASE_HEAD + """
    <div class="container py-4">
        <div class="row justify-content-center">
            <div class="col-lg-10">
                <div class="main-card">
                    <div class="d-flex justify-content-between align-items-center mb-4">
                        <div>
                            <h4 class="fw-bold mb-1">Analytics Dashboard</h4>
                            <p class="text-muted small mb-0">Biểu đồ xu hướng điểm chuẩn qua các năm</p>
                        </div>
                        <span class="badge bg-primary">2018 - 2025</span>
                    </div>
                    
                    <div style="height: 400px; width: 100%;">
                        <canvas id="trendChart"></canvas>
                    </div>

                    <div class="row mt-5 text-center">
                         {% for id, major in data.items() %}
                         <div class="col-md-4">
                            <div class="p-3 border rounded-3">
                                <h6 class="fw-bold text-truncate">{{ major.name }}</h6>
                                <p class="mb-0 text-success fw-bold">
                                    <i class="fas fa-arrow-up me-1"></i> Tăng trưởng TB
                                </p>
                            </div>
                         </div>
                         {% endfor %}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const chartData = {{ chart_json | safe }};
        const ctx = document.getElementById('trendChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData[0].years,
                datasets: Object.values(chartData).map((major, index) => ({
                    label: major.name,
                    data: major.scores,
                    borderColor: index === 0 ? '#4F46E5' : (index === 1 ? '#EC4899' : '#F59E0B'),
                    backgroundColor: 'rgba(0,0,0,0)',
                    tension: 0.4,
                    borderWidth: 3,
                    pointRadius: 4
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'top' } },
                scales: { y: { beginAtZero: false, min: 14 } }
            }
        });
    </script>
</body></html>
"""

# --- 3. FLASK ROUTES ---

@app.route('/')
def index():
    # Trang 1: Nhập liệu
    return render_template_string(PAGE_PREDICT.format(
        active_predict="active", active_result="", active_analytics="",
        majors=MAJORS
    ), majors=MAJORS)

@app.route('/process', methods=['POST'])
def process():
    # Xử lý Logic và Redirect sang trang Result
    try:
        major_id = int(request.form['major_id'])
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        tong_diem = toan + ly + hoa
        
        du_doan = 0
        if model:
            du_doan = model.predict([[2026, major_id]])[0]
            
        # Lưu kết quả vào SESSION (Bộ nhớ tạm của server cho user này)
        session['result'] = {
            'major_id': major_id,
            'tong_diem': round(tong_diem, 2),
            'diem_chuan': round(du_doan, 2)
        }
        return redirect(url_for('result'))
    except Exception as e:
        return f"Lỗi nhập liệu: {str(e)}"

@app.route('/result')
def result():
    # Trang 2: Kết quả (Lấy từ Session ra hiển thị)
    data = session.get('result')
    if not data:
        return render_template_string(PAGE_RESULT.format(
            active_predict="", active_result="active", active_analytics=""
        ), has_data=False)
    
    # Tính toán lại tỷ lệ để hiển thị
    chenh_lech = data['tong_diem'] - data['diem_chuan']
    if chenh_lech >= 1.5:
        ty_le, msg, color = 99, "CỰC KỲ AN TOÀN", "#059669" # Green
    elif chenh_lech >= 0:
        ty_le, msg, color = 85 + int(chenh_lech*5), "KHẢ QUAN", "#10B981" 
    elif chenh_lech >= -1.0:
        ty_le, msg, color = 50 + int(chenh_lech*30), "CẦN CÂN NHẮC", "#F59E0B" # Orange
    else:
        ty_le, msg, color = max(5, 50 + int(chenh_lech*20)), "RỦI RO CAO", "#EF4444" # Red
        
    return render_template_string(PAGE_RESULT.format(
        active_predict="", active_result="active", active_analytics=""
    ), has_data=True, 
       ten_nganh=MAJORS.get(data['major_id']),
       tong_diem=data['tong_diem'],
       diem_chuan=data['diem_chuan'],
       ty_le=int(ty_le),
       loi_khuyen=msg,
       color_code=color)

@app.route('/analytics')
def analytics():
    # Trang 3: Biểu đồ
    return render_template_string(PAGE_ANALYTICS.format(
        active_predict="", active_result="", active_analytics="active",
        chart_json=json.dumps(HISTORICAL_DATA)
    ), data=HISTORICAL_DATA, chart_json=json.dumps(HISTORICAL_DATA))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
