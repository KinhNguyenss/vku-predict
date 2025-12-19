from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model
model_path = 'vku_model.pkl'
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

# Giao diện HTML đẹp với Bootstrap
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dự đoán trúng tuyển VKU</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 50px; }
        .card { box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 15px; }
        .header-title { color: #d63384; font-weight: bold; }
        .btn-predict { background-color: #d63384; color: white; border: none; }
        .btn-predict:hover { background-color: #b02a6b; color: white; }
        .result-box { background-color: #e9ecef; padding: 20px; border-radius: 10px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card p-4">
                    <h2 class="text-center header-title mb-4">🔮 VKU AI PREDICTION</h2>
                    <p class="text-center text-muted">Nhập điểm thi để dự đoán khả năng trúng tuyển</p>
                    
                    <form method="post" action="/predict">
                        <div class="mb-3">
                            <label class="form-label">Điểm Toán</label>
                            <input type="number" step="0.1" min="0" max="10" class="form-control" name="toan" placeholder="Ví dụ: 8.5" required>
                        </div>
                        <div class="row">
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Điểm Lý</label>
                                <input type="number" step="0.1" min="0" max="10" class="form-control" name="ly" placeholder="Ví dụ: 7.0" required>
                            </div>
                            <div class="col-md-6 mb-3">
                                <label class="form-label">Điểm Hóa</label>
                                <input type="number" step="0.1" min="0" max="10" class="form-control" name="hoa" placeholder="Ví dụ: 7.5" required>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-bold">Điểm Chuyên Ngành (Hệ số 2)</label>
                            <input type="number" step="0.1" min="0" max="10" class="form-control" name="chuyen_nganh" placeholder="Ví dụ: 9.0 (Tin học, Tiếng Anh...)" required>
                        </div>
                        
                        <button type="submit" class="btn btn-predict w-100 py-2">🚀 Dự đoán ngay</button>
                    </form>

                    {% if ket_qua %}
                    <div class="result-box text-center">
                        <h4>Kết quả dự đoán:</h4>
                        <h1 class="display-4 fw-bold text-success">{{ ket_qua }}</h1>
                        <p>Tổng điểm xét tuyển dự kiến</p>
                    </div>
                    {% endif %}
                    
                    {% if loi %}
                    <div class="alert alert-danger mt-3">{{ loi }}</div>
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
    # Load lại model nếu chưa có (phòng trường hợp train xong sau khi app chạy)
    global model
    if model is None and os.path.exists(model_path):
        model = joblib.load(model_path)

    if not model:
        return render_template_string(HTML_TEMPLATE, loi="⚠️ Lỗi: Server chưa tìm thấy Model (vku_model.pkl). Hãy chạy Pipeline Training trước!")
    
    try:
        # Lấy dữ liệu từ form
        toan = float(request.form['toan'])
        ly = float(request.form['ly'])
        hoa = float(request.form['hoa'])
        cn = float(request.form['chuyen_nganh'])
        
        # Tạo DataFrame đúng chuẩn input của model
        input_data = pd.DataFrame({
            'Toan': [toan],
            'Ly': [ly],
            'Hoa': [hoa],
            'ChuyenNganh': [cn]
        })
        
        # Dự đoán
        prediction = model.predict(input_data)[0]
        
        return render_template_string(HTML_TEMPLATE, ket_qua=f"{prediction:.2f}")
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, loi=f"Có lỗi xảy ra: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
