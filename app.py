from flask import Flask, request, render_template_string
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load model
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Dictionary tên ngành để hiển thị
MAJORS = {
    0: "Công nghệ thông tin (Kỹ sư/Cử nhân)",
    1: "Quản trị Logistics & Chuỗi cung ứng số",
    2: "Quản trị kinh doanh"
}

# --- GIAO DIỆN HTML & CSS MỚI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự đoán trúng tuyển VKU 2026</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <linkcdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <style>
        :root {
            --primary-color: #6a11cb;
            --secondary-color: #2575fc;
            --accent-color: #ff4b2b;
            --success-color: #00b09b;
            --warning-color: #f7b733;
            --danger-color: #ff416c;
        }

        body {
            font-family: 'Poppins', sans-serif;
            /* Nền Gradient động */
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            min-height: 100vh;
            display: flex;
            align-items: center;
            padding: 40px 0;
        }

        /* Hiệu ứng Glassmorphism cho Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
            padding: 40px;
        }

        .app-title {
            font-weight: 800;
            background: linear-gradient(to right, var(--primary-color), var(--accent-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: 1px;
        }

        .form-label {
            font-weight: 600;
            color: #444;
            margin-bottom: 10px;
        }
        
        .form-select, .form-control {
            border-radius: 15px;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 4px rgba(37, 117, 252, 0.1);
        }

        .score-input {
            text-align: center;
            font-weight: 700;
            color: var(--primary-color);
        }

        .btn-predict {
            background: linear-gradient(to right, var(--accent-color), #ff416c);
            border: none;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 1px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .btn-predict:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 20px rgba(255, 75, 43, 0.4);
        }

        /* Phần kết quả */
        .result-section {
            margin-top: 40px;
            padding-top: 30px;
            border-top: 2px dashed #e0e0e0;
        }

        .score-box {
            background: #f8f9fa;
            border-radius: 20px;
            padding: 20px;
            text-align: center;
        }
        .score-box h3 { font-weight: 700; margin-bottom: 0; }

        /* Vòng tròn phần trăm (Circular Progress) */
        .progress-circle-container {
            display: flex;
            justify-content: center;
            margin: 30px 0;
        }
        .progress-circle {
            position: relative;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: conic-gradient(var(--color-status) var(--degree), #e0e0e0 0deg);
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.1);
        }
        .progress-circle::before {
            content: "";
            position: absolute;
            width: 140px;
            height: 140px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.95);
        }
        .progress-value {
            position: relative;
            font-size: 2.5rem;
            font-weight: 800;
            color: var(--color-status);
        }

        .advice-box {
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            background: var(--color-status);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="glass-card">
                    <div class="text-center mb-5">
                        <h1 class="app-title">🎓 VKU FUTURE PREDICT</h1>
                        <p class="text-muted">Dự đoán khả năng trúng tuyển Đại học 2026 bằng AI</p>
                    </div>
                    
                    <form method="post" action="/predict">
                        <div class="mb-4">
                            <label class="form-label"><i class="fas fa-graduation-cap me-2"></i>Chọn Ngành Mục Tiêu:</label>
                            <select name="major_id" class="form-select form-select-lg">
                                <option value="0">Công nghệ thông tin (HOT🔥)</option>
                                <option value="1">Logistics & Chuỗi cung ứng số (TOP🚀)</option>
                                <option value="2">Quản trị kinh doanh</option>
                            </select>
                        </div>
                        
                        <label class="form-label mb-3"><i class="fas fa-calculator me-2"></i>Nhập Điểm Thi Dự Kiến (3 Môn):</label>
                        <div class="row g-3 mb-4">
                            <div class="col-4">
                                <div class="input-group">
                                    <span class="input-group-text bg-white border-end-0"><i class="fas fa-square-root-alt text-primary"></i></span>
                                    <input type="number" step="0.1" min="0" max="10" name="toan" class="form-control score-input border-start-0" placeholder="Toán" required>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="input-group">
                                    <span class="input-group-text bg-white border-end-0"><i class="fas fa-atom text-success"></i></span>
                                    <input type="number" step="0.1" min="0" max="10" name="ly" class="form-control score-input border-start-0" placeholder="Lý" required>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="input-group">
                                    <span class="input-group-text bg-white border-end-0"><i class="fas fa-flask text-warning"></i></span>
                                    <input type="number" step="0.1" min="0" max="10" name="hoa" class="form-control score-input border-start-0" placeholder="Hóa/Anh" required>
                                </div>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary btn-predict w-100 py-3">
                            <i class="fas fa-rocket me-2"></i>DỰ ĐOÁN NGAY
                        </button>
                    </form>

                    {% if ket_qua %}
                    <div class="result-section" style="--color-status: {{ mau_sac }}; --degree: {{ degree_circle }}deg;">
                        <div class="text-center mb-4">
                            <h4 class="text-muted">Kết quả cho ngành:</h4>
                            <h3 style="color: var(--primary-color); font-weight: 700;">{{ ten_nganh }}</h3>
                        </div>

                        <div class="row g-3">
                            <div class="col-6">
                                <div class="score-box">
                                    <small class="text-muted display-block mb-2">Tổng điểm của bạn</small>
                                    <h3 style="color: var(--secondary-color);">{{ tong_diem }}</h3>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="score-box">
                                    <small class="text-muted display-block mb-2">Điểm chuẩn AI 2026</small>
                                    <h3 style="color: var(--accent-color);">~{{ diem_chuan_du_bao }}</h3>
                                </div>
                            </div>
                        </div>
                        
                        <div class="progress-circle-container">
                             <div class="progress-circle">
                                <span class="progress-value">{{ ty_le }}<small>%</small></span>
                            </div>
                        </div>

                        <div class="advice-box">
                            <h3><i class="fas fa-comment-dots me-2"></i>Lời khuyên từ AI</h3>
                            <p class="mb-0 fw-bold fs-5">{{ loi_khuyen }}</p>
                        </div>
                    </div>
                    {% endif %}
                    
                     {% if loi_he_thong %}
                    <div class="alert alert-danger mt-4 rounded-pill text-center">
                        <i class="fas fa-exclamation-triangle me-2"></i>{{ loi_he_thong }}
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    global model
    # Thử load model nếu chưa có
    if model is None:
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except:
                model = None
        
    if not model:
        return render_template_string(HTML_TEMPLATE, loi_he_thong="Chưa tìm thấy Model AI (vku_model.pkl). Vui lòng chạy Pipeline Training trước!")

    try:
        # 1. Lấy dữ liệu
        major_id = int(request.form['major_id'])
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        
        # 2. Tính toán tổng điểm của user
        tong_diem = toan + ly + hoa
        
        # 3. AI Dự đoán điểm chuẩn năm 2026 cho ngành đã chọn
        # Input cho model là 1 mảng 2 chiều: [[Năm, Mã Ngành]]
        du_doan_chuan = model.predict([[2026, major_id]])[0]
        
        # 4. Logic so sánh và tính tỷ lệ % đậu
        chenh_lech = tong_diem - du_doan_chuan
        
        # Màu sắc và lời khuyên dựa trên chênh lệch điểm
        if chenh_lech >= 1.5:
            ty_le = 99
            loi_khuyen = "Tuyệt vời! Tấm vé đại học gần như chắc chắn thuộc về bạn."
            mau_sac = "#00b09b" # Xanh ngọc (Success)
        elif chenh_lech >= 0.5:
            ty_le = 90 + int((chenh_lech - 0.5) * 5)
            loi_khuyen = "Cơ hội rất cao. Hãy giữ vững phong độ này nhé!"
            mau_sac = "#2ecc71" # Xanh lá
        elif chenh_lech >= -0.5:
            # Khoảng nguy hiểm: từ 80% xuống 50%
            ty_le = 50 + int((chenh_lech + 0.5) * 30)
            loi_khuyen = "Khá sát nút! Bạn đang ở ranh giới an toàn và nguy hiểm."
            mau_sac = "#f7b733" # Vàng cam (Warning)
        else:
            # Rất thấp: dưới 50%
            ty_le = max(5, 50 + int((chenh_lech + 0.5) * 20))
            loi_khuyen = "Cảnh báo! Mức điểm này rất khó cạnh tranh vào năm 2026."
            mau_sac = "#ff416c" # Đỏ hồng (Danger)

        # Tính độ phủ của vòng tròn tiến độ (3.6 độ = 1%)
        degree_circle = ty_le * 3.6

        return render_template_string(HTML_TEMPLATE, 
                                      ket_qua=True,
                                      ten_nganh=MAJORS.get(major_id, "Ngành khác"),
                                      tong_diem=round(tong_diem, 2),
                                      diem_chuan_du_bao=round(du_doan_chuan, 2),
                                      ty_le=int(ty_le),
                                      degree_circle=degree_circle, # Biến mới cho CSS vòng tròn
                                      loi_khuyen=loi_khuyen,
                                      mau_sac=mau_sac)

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, loi_he_thong=f"Có lỗi dữ liệu đầu vào: {str(e)}. Vui lòng nhập số hợp lệ.")

if __name__ == '__main__':
    # Chạy app ở tất cả các IP, cổng 5000, chế độ debug tắt khi deploy thật
    app.run(host='0.0.0.0', port=5000, debug=False)
