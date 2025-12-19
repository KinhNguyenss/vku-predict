import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# 1. Tạo Dataset từ dữ liệu lịch sử (Dựa trên các ảnh bạn gửi)
# Tôi lấy đại diện 3 ngành HOT nhất để demo: CNTT, Marketing, QT Logistics
data = {
    'Year': [
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, # CNTT
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, # Logistics
        2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025  # QTKD
    ],
    'Major_ID': [
        0, 0, 0, 0, 0, 0, 0, 0, # 0 là CNTT
        1, 1, 1, 1, 1, 1, 1, 1, # 1 là Logistics
        2, 2, 2, 2, 2, 2, 2, 2  # 2 là QTKD
    ],
    'Score': [
        17.0, 19.75, 18.0, 23.0, 25.0, 24.0, 24.0, 24.71, # Điểm CNTT qua các năm
        16.0, 18.0, 18.0, 23.0, 25.0, 23.0, 26.0, 26.3,   # Điểm Logistics
        16.0, 18.75, 18.0, 22.5, 24.0, 23.0, 25.0, 25.65  # Điểm QTKD
    ]
}

df = pd.DataFrame(data)

# 2. Huấn luyện Model Hồi quy tuyến tính (Linear Regression)
# Model sẽ học xu hướng tăng điểm theo thời gian (Year) và sự khác biệt giữa các ngành (Major)
X = df[['Year', 'Major_ID']]
y = df['Score']

print("Dang huan luyen model du doan diem chuan...")
model = LinearRegression()
model.fit(X, y)

# 3. Lưu Model
joblib.dump(model, 'vku_model.pkl')
print("Model da duoc luu tai vku_model.pkl")

# Test thử dự đoán năm 2026 cho ngành CNTT (ID=0)
pred_2026 = model.predict([[2026, 0]])
print(f"Du doan diem chuan CNTT nam 2026: {pred_2026[0]:.2f}")
