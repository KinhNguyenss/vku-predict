import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# 1. Load dữ liệu
print("Dang doc du lieu tu dataset.csv...")
df = pd.read_csv('dataset.csv')

# 2. Chuẩn bị dữ liệu (Lấy năm làm feature, điểm làm target)
X = df[['Year']]
y = df['Score']

# 3. Huấn luyện model (Hồi quy tuyến tính)
print("Dang huan luyen model...")
model = LinearRegression()
model.fit(X, y)

# 4. Lưu model
print(f"Model da duoc huan luyen! Du doan diem nam 2025: {model.predict([[2025]])[0]:.2f}")
joblib.dump(model, 'vku_model.pkl')
print("Da luu model tai: vku_model.pkl")
