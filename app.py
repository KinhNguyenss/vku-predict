from flask import Flask, request, render_template_string
import joblib
import os
import json
import numpy as np

app = Flask(__name__)

# --- 1. CONFIG & DATA ---
# Load model
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Dữ liệu mẫu lịch sử điểm chuẩn (Lấy từ hình ảnh bạn cung cấp để vẽ biểu đồ)
# Đây là dữ liệu thật để hiển thị tab Analytics
HISTORICAL_DATA = {
    0: { # CNTT
        "name": "Công nghệ thông tin",
        "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "scores": [17.0, 19.75, 18.0, 23.0, 25.0, 23.5, 24.0, 24.71]
    },
    1: { # Logistics
        "name": "Logistics & Chuỗi cung ứng",
        "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "scores": [16.0, 18.0, 18.0, 23.0, 25.0, 23.0, 26.0, 26.3]
    },
    2: { # QTKD
        "name": "Quản trị kinh doanh",
        "years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "scores": [16.0, 18.75, 18.0, 22.5, 24.0, 23.0, 25.0, 25.65]
    }
}

MAJORS = {k: v["name"] for k, v in HISTORICAL_DATA.items()}

# --- 2. GIAO DIỆN HTML/CSS/JS ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VKU AI Prediction Platform</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --primary: #4F46E5; /* Indigo */
            --secondary: #7C3AED; /* Purple */
            --bg-color: #F3F4F6;
            --text-dark: #1F2937;
            --card-bg: #FFFFFF;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-dark);
        }

        /* Header Style */
        .navbar {
            background: var(--card-bg);
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 1rem 2rem;
        }
        .navbar-brand {
            font-weight: 700;
            font-size: 1.5rem;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Navigation Tabs (Pills) */
        .nav-pills .nav-link {
            color: #6B7280;
            font-weight: 600;
            border-radius: 50px;
            padding: 10px 25px;
            margin-right: 10px;
            transition: all 0.3s;
        }
        .nav-pills .nav-link.active {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            box-shadow: 0 4px 6px rgba(79, 70, 229, 0.3);
        }

        /* Cards */
        .main-card {
            background: var(--card-bg);
            border-radius: 16px;
            border: 1px solid #E5E7EB;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            padding: 30px;
            margin-top: 20px;
            min-height: 500px;
        }

        .history-card {
            background: #F9FAFB;
            border-left: 1px solid #E5E7EB;
            padding: 20px;
            height: 100%;
            border-radius: 0 16px 16px 0;
        }

        /* Form Elements */
        .form-label { font-weight: 600; font-size: 0.9rem; color: #374151; }
        .form-control, .form-select {
            border-radius: 10px;
            border: 1px solid #D1D5DB;
            padding: 12px;
        }
        .form-control:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .btn-generate {
            background: linear-gradient(to right, #111827, #374151);
            color: white;
            border-radius: 10px;
            font-weight: 600;
            padding: 12px;
            width: 100%;
            transition: transform 0.2s;
        }
        .btn-generate:hover { transform: translateY(-2px); color: white;}

        /* Result Section */
        .score-display {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .history-item {
            background: white;
            padding: 10px;
            border-radius: 8px;
            margin-bottom: 10px;
            border: 1px solid #E5E7EB;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>

    <nav class="navbar navbar-expand-lg sticky-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="#"><i class="fas fa-brain me-2"></i>VKU AI Platform</a>
        </div>
    </nav>

    <div class="container py-4">
        
        <ul class="nav nav-pills justify-content-center mb-3" id="pills-tab" role="tablist">
            <li class="nav-item">
                <button class="nav-link {{ 'active' if active_tab == 'prediction' else '' }}" id="tab-prediction" onclick="switchTab('prediction')">
                    <i class="fas fa-plus-circle me-2"></i>New Prediction
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link {{ 'active' if active_tab == 'result' else '' }}" id="tab-result" onclick="switchTab('result')" {{ 'disabled' if not ket_qua else '' }}>
                    <i class="fas fa-poll-h me-2"></i>Results
                </button>
            </li>
            <li class="nav-item">
                <button class="nav-link" id="tab-analytics" onclick="switchTab('analytics')">
                    <i class="fas fa-chart-line me-2"></i>Analytics
                </button>
            </li>
        </ul>

        <div class="main-card position-relative">
            <div class="row h-100">
                
                <div id="section-prediction" class="col-md-8 fade-in" style="display: {{ 'block' if active_tab == 'prediction' else 'none' }};">
                    <h4 class="mb-4"><i class="fas fa-robot me-2 text-primary"></i>AI Prediction Engine</h4>
                    <form method="post" action="/predict">
                        <div class="mb-4">
                            <label class="form-label">Chọn Chuyên Ngành</label>
                            <select name="major_id" class="form-select bg-light">
                                {% for id, name in majors.items() %}
                                    <option value="{{ id }}">{{ name }}</option>
                                {% endfor %}
                            </select>
                        </div>

                        <div class="row g-3 mb-4">
                            <div class="col-md-4">
                                <label class="form-label">Điểm Toán</label>
                                <input type="number" step="0.1" name="toan" class="form-control" placeholder="0.0 - 10.0" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Điểm Lý</label>
                                <input type="number" step="0.1" name="ly" class="form-control" placeholder="0.0 - 10.0" required>
                            </div>
                            <div class="col-md-4">
                                <label class="form-label">Điểm Hóa/Anh</label>
                                <input type="number" step="0.1" name="hoa" class="form-control" placeholder="0.0 - 10.0" required>
                            </div>
                        </div>

                        <div class="mb-4">
                             <label class="form-label">Năm Dự Đoán</label>
                             <select class="form-select" disabled>
                                 <option>2026 (AI Future Forecast)</option>
                             </select>
                        </div>

                        <button type="submit" class="btn btn-generate">
                            <i class="fas fa-magic me-2"></i>Generate Prediction
                        </button>
                    </form>
                </div>

                <div id="section-result" class="col-md-8 text-center d-flex flex-column justify-content-center" style="display: {{ 'block' if active_tab == 'result' else 'none' }};">
                    {% if ket_qua %}
                        <div class="mb-3">
                            <i class="fas fa-check-circle text-success fa-3x mb-3"></i>
                            <h2 class="text-muted">Kết quả dự đoán cho ngành</h2>
                            <h4 class="fw-bold">{{ ten_nganh }}</h4>
                        </div>

                        <div class="row justify-content-center my-4">
                            <div class="col-md-5">
                                <div class="p-3 bg-light rounded-3 border">
                                    <small class="text-muted">Tổng điểm của bạn</small>
                                    <div class="fs-2 fw-bold text-dark">{{ tong_diem }}</div>
                                </div>
                            </div>
                            <div class="col-md-5">
                                <div class="p-3 bg-light rounded-3 border">
                                    <small class="text-muted">Điểm chuẩn AI (2026)</small>
                                    <div class="fs-2 fw-bold text-primary">{{ diem_chuan_du_bao }}</div>
                                </div>
                            </div>
                        </div>

                        <div class="my-4">
                            <h5 class="text-muted mb-3">Tỷ lệ đậu dự kiến</h5>
                            <div class="score-display">{{ ty_le }}%</div>
                            <p class="mt-2 badge bg-{{ 'success' if ty_le > 80 else 'warning' if ty_le > 50 else 'danger' }} fs-6 px-3 py-2">
                                {{ loi_khuyen }}
                            </p>
                        </div>
                        
                        <button onclick="switchTab('prediction')" class="btn btn-outline-secondary mt-3">
                            <i class="fas fa-arrow-left me-2"></i>Dự đoán lại
                        </button>

                        <script>
                            window.onload = function() {
                                saveHistory("{{ ten_nganh }}", "{{ tong_diem }}", "{{ ty_le }}");
                            };
                        </script>
                    {% else %}
                        <div class="text-muted">
                            <i class="fas fa-wind fa-3x mb-3"></i>
                            <p>Chưa có dữ liệu dự đoán. Hãy nhập thông tin bên tab New Prediction.</p>
                        </div>
                    {% endif %}
                </div>

                <div id="section-analytics" class="col-md-8" style="display: none;">
                    <h4 class="mb-4"><i class="fas fa-chart-area me-2 text-primary"></i>Phân Tích Xu Hướng Điểm Chuẩn</h4>
                    <p class="text-muted small">Dữ liệu tổng hợp từ 2018 đến 2025</p>
                    <canvas id="trendChart" height="200"></canvas>
                </div>

                <div class="col-md-4 border-start">
                    <div class="ps-3">
                        <h5 class="mb-3 fw-bold"><i class="fas fa-history me-2 text-secondary"></i>Lịch sử dự đoán</h5>
                        <div id="history-list" class="overflow-auto" style="max-height: 400px;">
                            <div class="text-center text-muted mt-5">
                                <small>Chưa có lịch sử</small>
                            </div>
                        </div>
                        <button onclick="clearHistory()" class="btn btn-sm btn-light text-danger w-100 mt-3">
                            <i class="fas fa-trash me-2"></i>Xóa lịch sử
                        </button>
                    </div>
                </div>

            </div>
        </div>
    </div>

    <script>
        // 1. Logic chuyển Tab
        function switchTab(tabName) {
            // Ẩn tất cả section
            document.getElementById('section-prediction').style.display = 'none';
            document.getElementById('section-result').style.display = 'none';
            document.getElementById('section-analytics').style.display = 'none';
            
            // Bỏ active class ở nav
            document.querySelectorAll('.nav-link').forEach(el => el.classList.remove('active'));

            // Hiện section được chọn
            document.getElementById('section-' + tabName).style.display = 'block';
            document.getElementById('tab-' + tabName).classList.add('active');
        }

        // 2. Logic Vẽ Biểu Đồ (Analytics)
        // Nhận dữ liệu từ Python
        const chartData = {{ chart_json | safe }};
        
        const ctx = document.getElementById('trendChart').getContext('2d');
        const myChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: chartData[0].years, // Lấy mốc năm từ ngành đầu tiên
                datasets: Object.values(chartData).map((major, index) => ({
                    label: major.name,
                    data: major.scores,
                    borderColor: index === 0 ? '#4F46E5' : (index === 1 ? '#7C3AED' : '#F59E0B'),
                    backgroundColor: 'rgba(0,0,0,0)',
                    tension: 0.4,
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom' }
                },
                scales: {
                    y: { beginAtZero: false, min: 15, max: 30 }
                }
            }
        });

        // 3. Logic LocalStorage History
        function saveHistory(nganh, diem, tyle) {
            let history = JSON.parse(localStorage.getItem('vku_history')) || [];
            const newRecord = {
                nganh: nganh,
                diem: diem,
                tyle: tyle,
                time: new Date().toLocaleTimeString()
            };
            // Thêm vào đầu danh sách
            history.unshift(newRecord);
            // Giới hạn 10 bản ghi
            if(history.length > 10) history.pop();
            
            localStorage.setItem('vku_history', JSON.stringify(history));
            renderHistory();
        }

        function renderHistory() {
            let history = JSON.parse(localStorage.getItem('vku_history')) || [];
            const container = document.getElementById('history-list');
            
            if (history.length === 0) {
                container.innerHTML = '<div class="text-center text-muted mt-5"><small>Chưa có lịch sử</small></div>';
                return;
            }

            let html = '';
            history.forEach(item => {
                let badgeColor = item.tyle > 80 ? 'success' : (item.tyle > 50 ? 'warning' : 'danger');
                html += `
                    <div class="history-item shadow-sm">
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="fw-bold text-primary" style="font-size: 0.8rem">${item.nganh}</span>
                            <span class="text-muted" style="font-size: 0.7rem">${item.time}</span>
                        </div>
                        <div class="mt-2 d-flex justify-content-between">
                            <span>Điểm: <b>${item.diem}</b></span>
                            <span class="badge bg-${badgeColor}">${item.tyle}%</span>
                        </div>
                    </div>
                `;
            });
            container.innerHTML = html;
        }

        function clearHistory() {
            localStorage.removeItem('vku_history');
            renderHistory();
        }

        // Render lịch sử khi tải trang
        renderHistory();

    </script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    # Mặc định vào tab prediction
    return render_template_string(HTML_TEMPLATE, 
                                  majors=MAJORS, 
                                  active_tab='prediction', 
                                  ket_qua=None,
                                  chart_json=json.dumps(HISTORICAL_DATA))

@app.route('/predict', methods=['POST'])
def predict():
    global model
    if model is None and os.path.exists(model_path):
        model = joblib.load(model_path)
    
    # Mặc định giá trị trả về
    active_tab = 'result'
    ket_qua_data = None
    
    try:
        # Lấy dữ liệu
        major_id = int(request.form['major_id'])
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        
        # Logic tính toán
        tong_diem = toan + ly + hoa
        
        if model:
            du_doan_chuan = model.predict([[2026, major_id]])[0]
        else:
            du_doan_chuan = 0 # Fallback nếu không có model
            
        chenh_lech = tong_diem - du_doan_chuan
        
        # Logic tính tỷ lệ (như cũ)
        if chenh_lech >= 2.0:
            ty_le = 99
            loi_khuyen = "Safe Zone - Đậu chắc!"
        elif chenh_lech >= 0:
            ty_le = 80 + (chenh_lech * 10)
            loi_khuyen = "High Chance - Khả năng cao"
        elif chenh_lech >= -1.0:
            ty_le = 50 + (chenh_lech * 30)
            loi_khuyen = "Risky - Cần cân nhắc"
        else:
            ty_le = max(0, 50 + (chenh_lech * 10))
            loi_khuyen = "Hard - Rất khó"

        return render_template_string(HTML_TEMPLATE, 
                                      majors=MAJORS,
                                      active_tab='result',
                                      ket_qua=True,
                                      ten_nganh=MAJORS.get(major_id),
                                      tong_diem=round(tong_diem, 2),
                                      diem_chuan_du_bao=round(du_doan_chuan, 2),
                                      ty_le=int(ty_le),
                                      loi_khuyen=loi_khuyen,
                                      chart_json=json.dumps(HISTORICAL_DATA))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
