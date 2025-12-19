import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# 1. Tạo Dataset chuẩn từ 2021 - 2025 (Dữ liệu từ PDF & Ảnh)
# ID các ngành:
# 0: CNTT
# 1: AI (Trí tuệ nhân tạo)
# 2: An toàn thông tin
# 3: Thiết kế vi mạch / Máy tính
# 4: Marketing
# 5: Logistics
# 6: Quản trị kinh doanh (QTKD)
# 7: Du lịch & Lữ hành số
# 8: Tài chính số / Công nghệ tài chính

data = {
    'Year': [],
    'Major_ID': [],
    'Score': []
}

# Dữ liệu chi tiết từng năm (Year, Major_ID, Score)
# Nguon: PDF & Image user provided
records = [
    # --- CNTT (ID: 0) ---
    (2021, 0, 23.0), (2022, 0, 25.0), (2023, 0, 25.01), (2024, 0, 25.0), (2025, 0, 20.0),
    
    # --- AI (ID: 1) ---
    (2021, 1, 21.05), (2022, 1, 25.0), (2023, 1, 25.01), (2024, 1, 25.0), (2025, 1, 21.0),

    # --- An toàn thông tin (ID: 2) ---
    (2021, 2, 22.0), (2022, 2, 24.5), (2023, 2, 23.0), (2024, 2, 24.0), (2025, 2, 19.0),

    # --- Kỹ thuật máy tính / Vi mạch (ID: 3) ---
    (2021, 3, 20.0), (2022, 3, 24.0), (2023, 3, 23.0), (2024, 3, 27.0), (2025, 3, 24.0),

    # --- Marketing (ID: 4) ---
    (2021, 4, 22.0), (2022, 4, 25.0), (2023, 4, 23.0), (2024, 4, 26.0), (2025, 4, 23.25),

    # --- Logistics (ID: 5) ---
    (2021, 5, 23.0), (2022, 5, 25.0), (2023, 5, 23.0), (2024, 5, 26.0), (2025, 5, 23.5),

    # --- QTKD (ID: 6) ---
    (2021, 6, 22.5), (2022, 6, 24.0), (2023, 6, 23.0), (2024, 6, 25.0), (2025, 6, 22.0),
    
    # --- Du lịch (ID: 7) ---
    (2021, 7, 20.5), (2022, 7, 24.0), (2023, 7, 22.5), (2024, 7, 25.0), (2025, 7, 23.0),

    # --- Tài chính (ID: 8) ---
    (2021, 8, 21.0), (2022, 8, 24.0), (2023, 8, 22.5), (2024, 8, 24.0), (2025, 8, 22.0),
]

for r in records:
    data['Year'].append(r[0])
    data['Major_ID'].append(r[1])
    data['Score'].append(r[2])

df = pd.DataFrame(data)

# 2. Huấn luyện Model
# Input: Năm + ID Ngành -> Output: Điểm chuẩn
X = df[['Year', 'Major_ID']]
y = df['Score']

print("Dang huan luyen model voi du lieu 2021-2025...")
model = LinearRegression()
model.fit(X, y)

# 3. Lưu Model
joblib.dump(model, 'vku_model.pkl')
print("Model da duoc luu tai vku_model.pkl")
