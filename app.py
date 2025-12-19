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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự đoán trúng tuyển VKU 2026</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); min-height: 100vh; padding-top: 50px; }
        .card { border: none; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.2); }
        .btn-predict { background-color: #ff6b6b; color: white; border: none; font-size: 1.2rem; }
        .btn-predict:hover { background-color: #ee5253; }
        .score-input { text-align: center; font-size: 1.2rem; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card p-5">
                    <h2 class="text-center mb-4 text-primary fw-bold">🎓 VKU ADMISSION PREDICTOR 2026</h2>
                    <p class="text-center text-muted mb-4">Nhập điểm thi THPT Quốc gia dự kiến của bạn</p>
                    
                    <form method="post" action="/predict">
                        <div class="mb-4">
                            <label class="form-label fw-bold">Chọn Ngành Muốn Xét Tuyển:</label>
                            <select name="major_id" class="form-select form-select-lg">
                                <option value="0">Công nghệ thông tin</option>
                                <option value="1">Logistics & Chuỗi cung ứng</option>
                                <option value="2">Quản trị kinh doanh</option>
                            </select>
                        </div>
                        
                        <div class="row mb-4">
                            <div class="col-4">
                                <label class="form-label text-center w-100">Toán</label>
                                <input type="number" step="0.1" min="0" max="10" name="toan" class="form-control score-input" required>
                            </div>
                            <div class="col-4">
                                <label class="form-label text-center w-100">Lý</label>
                                <input type="number" step="0.1" min="0" max="10" name="ly" class="form-control score-input" required>
                            </div>
                            <div class="col-4">
                                <label class="form-label text-center w-100">Hóa/Anh</label>
                                <input type="number" step="0.1" min="0" max="10" name="hoa" class="form-control score-input" required>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-predict w-100 py-3 rounded-pill">🔮 Dự Đoán Tỷ Lệ Đậu</button>
                    </form>

                    {% if ket_qua %}
                    <hr class="my-4">
                    <div class="text-center">
                        <h4>Ngành: <span class="text-info">{{ ten_nganh }}</span></h4>
                        <div class="row mt-3">
                            <div class="col-6">
                                <p class="mb-1">Tổng điểm của bạn</p>
                                <h3 class="text-primary">{{ tong_diem }}</h3>
                            </div>
                            <div class="col-6">
                                <p class="mb-1">Điểm chuẩn dự báo 2026</p>
                                <h3 class="text-danger">{{ diem_chuan_du_bao }}</h3>
                            </div>
                        </div>
                        
                        <div class="mt-3 p-3 rounded" style="background-color: {{ mau_sac }}; color: white;">
                            <h3>Tỷ lệ đậu: {{ ty_le }}%</h3>
                            <p class="mb-0 fw-bold">{{ loi_khuyen }}</p>
                        </div>
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
    if model is None and os.path.exists(model_path):
        model = joblib.load(model_path)
    
    if not model:
        return "Chưa có Model! Hãy chạy Training trước."

    try:
        # 1. Lấy dữ liệu
        major_id = int(request.form['major_id'])
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        
        # 2. Tính toán
        tong_diem = toan + ly + hoa
        
        # 3. Dự đoán điểm chuẩn 2026
        # Input cho model phải là [[Nam, Ma_Nganh]]
        du_doan_chuan = model.predict([[2026, major_id]])[0]
        
        # 4. Tính tỷ lệ đậu (Logic đơn giản)
        chenh_lech = tong_diem - du_doan_chuan
        
        if chenh_lech >= 2.0:
            ty_le = 99
            loi_khuyen = "Chúc mừng! Vé vào đại học nằm trong tay bạn."
            mau_sac = "#2ecc71" # Xanh la
        elif chenh_lech >= 0:
            ty_le = 80 + (chenh_lech * 10)
            loi_khuyen = "Cơ hội rất cao, nhưng đừng chủ quan!"
            mau_sac = "#27ae60"
        elif chenh_lech >= -1.0:
            ty_le = 50 + (chenh_lech * 30) # Giam dan
            loi_khuyen = "Khá nguy hiểm! Cần cố gắng thêm chút nữa."
            mau_sac = "#f39c12" # Vang
        else:
            ty_le = max(0, 50 + (chenh_lech * 10))
            loi_khuyen = "Rất khó đậu. Hãy cân nhắc nguyện vọng khác hoặc nỗ lực vượt bậc!"
            mau_sac = "#e74c3c" # Do

        return render_template_string(HTML_TEMPLATE, 
                                      ket_qua=True,
                                      ten_nganh=MAJORS.get(major_id, "Unknown"),
                                      tong_diem=round(tong_diem, 2),
                                      diem_chuan_du_bao=round(du_doan_chuan, 2),
                                      ty_le=int(ty_le),
                                      loi_khuyen=loi_khuyen,
                                      mau_sac=mau_sac)

    except Exception as e:
        return f"Có lỗi: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
