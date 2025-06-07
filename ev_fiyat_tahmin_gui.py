import pandas as pd
import joblib
import tkinter as tk
from tkinter import messagebox
from math import sqrt
from sklearn.metrics import mean_squared_error

# 🔹 Veriyi oku
df = pd.read_csv("train.csv")

# 🔹 Kategorik sütunları sayısala çevir
for col in df.select_dtypes(include='object').columns:
    df[col] = pd.factorize(df[col])[0]

df.fillna(df.mean(numeric_only=True), inplace=True)

# 🔹 Türkçeleştirilmiş özellikler
ozellikler = {
    'OverallQual': 'Genel Kalite (1-10)',
    'GrLivArea': 'Yaşanabilir Alan (sqft)',
    'GarageCars': 'Garaj Kapasitesi (araç)',
    'GarageArea': 'Garaj Alanı (sqft)',
    'TotalBsmtSF': 'Bodrum Alanı (sqft)',
    'FullBath': 'Tam Banyo Sayısı',
    'YearBuilt': 'İnşa Yılı'
}

df = df[list(ozellikler.keys()) + ['SalePrice']]
X = df[list(ozellikler.keys())]
y = df['SalePrice']

# 🔹 Modeli yükle
model = joblib.load("model.pkl")

# 🔹 RMSE ve R² hesaplama
y_pred = model.predict(X)
rmse = sqrt(mean_squared_error(y, y_pred))
r2 = model.score(X, y)

# 🔹 Tahmin fonksiyonu
def tahmin_et():
    try:
        girdiler = [float(girisler[tr_ozellik].get()) for tr_ozellik in ozellikler.values()]
        tahmin = model.predict([girdiler])[0]
        # Tahmin değerini GUI üzerinde güncelle
        tahmin_label.config(text=f"Tahmini Ev Fiyatı: {round(tahmin, 2)} USD")
    except Exception as e:
        messagebox.showerror("Hata", f"Girişlerde sorun var: {e}")

# 🔹 Arayüz
pencere = tk.Tk()
pencere.title("🏠 Ev Fiyat Tahmini Uygulaması")
pencere.geometry("460x620")  # Yükseklik biraz arttırıldı
pencere.configure(bg="#f1f1f1")

# 🔸 Logo
try:
    from PIL import Image, ImageTk
    logo_img = Image.open("logo.png")
    logo_img = logo_img.resize((200, 80), Image.ANTIALIAS)
    logo = ImageTk.PhotoImage(logo_img)
    tk.Label(pencere, image=logo, bg="#f1f1f1").pack(pady=10)
except:
    tk.Label(pencere, text="🏠 Emlak Tahmin Sistemi", font=("Helvetica", 16, "bold"), bg="#f1f1f1").pack(pady=10)

# 🔸 Giriş Alanları
form_frame = tk.Frame(pencere, bg="#f1f1f1")
form_frame.pack(pady=5)

girisler = {}
for i, (ozet, tr_ozellik) in enumerate(ozellikler.items()):
    tk.Label(form_frame, text=tr_ozellik, bg="#f1f1f1", font=("Arial", 10)).grid(row=i, column=0, sticky="w", padx=5, pady=6)
    entry = tk.Entry(form_frame, width=25, font=("Arial", 10))
    entry.grid(row=i, column=1, padx=10)
    girisler[tr_ozellik] = entry

# 🔸 Buton ve Başarı Skoru
tk.Button(pencere, text="💰 Fiyatı Tahmin Et", command=tahmin_et, bg="#4CAF50", fg="white",
          font=("Arial", 11, "bold"), padx=10, pady=5).pack(pady=20)

# 🔸 Model Başarı Oranı (R²) ve RMSE
tk.Label(pencere, text=f"R²: {round(r2, 4)}", bg="#f1f1f1", font=("Arial", 10, "italic")).pack()
tk.Label(pencere, text=f"RMSE: {round(rmse, 2)}", bg="#f1f1f1", font=("Arial", 10, "italic")).pack()

# 🔸 Tahmin Sonucu için Etiket
tahmin_label = tk.Label(pencere, text="Tahmini Ev Fiyatı: ", bg="#f1f1f1", font=("Arial", 12, "bold"))
tahmin_label.pack(pady=20)

# Arayüzü başlat
pencere.mainloop()
