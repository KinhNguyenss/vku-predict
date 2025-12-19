import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import numpy as np

# 1. Tạo dữ liệu giả lập (Thay vì đọc file csv chưa có chuẩn)
# Giả sử công thức: Điểm chuẩn = Toán + Lý + Hóa + (Chuyên ngành * 2) + Nhiễu
print("Dang tao du lieu mau...")
n_samples = 1000
np.random.seed(42)

# Tạo điểm ngẫu nhiên từ 5 đến 10
toan = np.random.uniform(5, 10, n_samples)
ly = np.random.uniform(5, 10, n_samples)
hoa = np.random.uniform(5, 10, n_samples)
chuyen_nganh = np.random.uniform(5, 10, n_samples)

# Tạo DataFrame
X = pd.DataFrame({
    'Toan': toan, 
    'Ly': ly, 
    'Hoa': hoa, 
    'ChuyenNganh': chuyen_nganh
})

# Giả lập điểm chuẩn trúng tuyển (Tổng điểm + chút sai số ngẫu nhiên)
# Ví dụ: VKU xét tuyển hệ số 2 môn chuyên ngành
y = toan + ly + hoa + (chuyen_nganh * 2) + np.random.normal(0, 0.5, n_samples)

# 2. Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Huấn luyện Model
print("Dang huan luyen model Linear Regression...")
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Đánh giá nhanh
score = model.score(X_test, y_test)
print(f"Do chinh xac cua model: {score:.2f}")

# 5. Lưu Model
joblib.dump(model, 'vku_model.pkl')
print("Da luu model tai: vku_model.pkl")
