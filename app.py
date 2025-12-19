from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load model (kiểm tra file tồn tại không)
model_path = 'vku_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None

# Giao diện đơn giản (HTML nằm trong code cho gọn)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dự đoán trúng tuyển VKU</title>
</head>
<body>
    <h2>Hệ thống dự đoán trúng tuyển VKU</h2>
    <form method="post" action="/predict">
        <label>Nhập điểm thi:</label><br>
        <input type="text" name="diem" required><br><br>
        <input type="submit" value="Dự đoán">
    </form>
    {% if ket_qua %}
        <h3>Kết quả: {{ ket_qua }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template_string(HTML_TEMPLATE, ket_qua="Lỗi: Chưa có Model!")
    
    try:
        diem = float(request.form['diem'])
        # Lưu ý: Model của bạn train với dữ liệu nào thì input phải y hệt.
        # Ở đây tôi ví dụ input là 1 mảng 2 chiều theo chuẩn scikit-learn
        # Nếu model cần nhiều cột hơn (như ngành, khu vực), bạn cần thêm input.
        prediction = model.predict([[diem]]) 
        ket_qua_text = f"Dự đoán điểm chuẩn năm tới: {prediction[0]:.2f}"
    except Exception as e:
        ket_qua_text = f"Có lỗi xảy ra: {str(e)}"

    return render_template_string(HTML_TEMPLATE, ket_qua=ket_qua_text)

if __name__ == '__main__':
    # Chạy ở cổng 5000
    app.run(host='0.0.0.0', port=5000)
