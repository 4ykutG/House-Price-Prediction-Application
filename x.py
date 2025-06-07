import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from scipy import stats

# 🔹 Veri setini oku
data = pd.read_csv("train.csv")  # Veri setinizi buraya ekleyin

# 🔹 Kategorik sütunları sayısallaştırma
for col in data.select_dtypes(include='object').columns:
    data[col] = pd.factorize(data[col])[0]

# 🔹 Eksik verileri doldurma (ortalama ile)
imputer = SimpleImputer(strategy="mean")
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# 🔹 Özellikleri ve hedef değişkeni ayırma
X = data.drop(columns=["SalePrice"])  # Özellikler
y = data["SalePrice"]  # Hedef değişken

# 🔹 Aykırı değer tespiti (Z-skoru yöntemiyle)
z_scores = stats.zscore(X)
X = X[(z_scores < 3).all(axis=1)]  # Z-skoru 3'ten büyük olanları çıkar

# 🔹 Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Özellikleri ölçeklendirme (normalizasyon)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Modeli eğitme
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 🔹 Modeli kaydetme
joblib.dump(model, "model.pkl")

# 🔹 Modelin doğruluğunu kontrol etme
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")
