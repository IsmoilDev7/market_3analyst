import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mahsulot Zarar Analizi", layout="wide")

st.title("📊 Mahsulotlar bo‘yicha zarar / foyda analitikasi")

# =========================
# 1. EXCEL YUKLASH
# =========================
file = st.file_uploader("📂 Sotuv / Qaytish Excel faylni yuklang", type=["xlsx", "xls"])

if not file:
    st.info("Excel fayl yuklang")
    st.stop()

df = pd.read_excel(file)

# =========================
# 2. MAJBURIY USTUNLAR
# =========================
required_cols = [
    "Период",
    "Номенклатура",
    "Количество",
    "Продажная сумма",
    "Возврат сумма"
]

for col in required_cols:
    if col not in df.columns:
        st.error(f"❌ '{col}' ustuni topilmadi")
        st.stop()

# =========================
# 3. DATA TAYYORLASH
# =========================
df["Период"] = pd.to_datetime(df["Период"], errors="coerce")
df = df.dropna(subset=["Период"])

df["day"] = df["Период"].dt.date

# =========================
# 4. SANA FILTRI
# =========================
c1, c2 = st.columns(2)
date_from = c1.date_input("Boshlanish sana", df["Период"].min())
date_to   = c2.date_input("Tugash sana", df["Период"].max())

df = df[
    (df["Период"] >= pd.to_datetime(date_from)) &
    (df["Период"] <= pd.to_datetime(date_to))
]

# =========================
# 5. KUNLIK MAHSULOT ANALIZI
# =========================
daily = df.groupby(
    ["day", "Номенклатура"], as_index=False
).agg(
    sold_qty=("Количество", "sum"),
    sales_sum=("Продажная сумма", "sum"),
    return_sum=("Возврат сумма", "sum")
)

daily["loss_percent"] = np.where(
    daily["sales_sum"] > 0,
    (daily["return_sum"] / daily["sales_sum"]) * 100,
    0
).clip(0, 100)

# =========================
# 6. ZARAR / FOYDA LABEL
# =========================
daily["label"] = np.where(daily["loss_percent"] > 20, 1, 0)
# 1 = ZARAR, 0 = FOYDA

# =========================
# 7. ML MODEL
# =========================
X = daily[["sold_qty", "sales_sum", "return_sum", "loss_percent"]]
y = daily["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# =========================
# 8. BASHORAT
# =========================
daily["prediction"] = model.predict(X)
daily["Natija"] = daily["prediction"].map({
    1: "❌ ZARAR keltiradi",
    0: "✅ FOYDA keltiradi"
})

# =========================
# 9. JADVAL
# =========================
st.subheader("📋 Kunlik mahsulotlar bo‘yicha natija")
st.dataframe(
    daily.sort_values("loss_percent", ascending=False),
    use_container_width=True
)

# =========================
# 10. KPI
# =========================
st.subheader("📌 Umumiy ko‘rsatkichlar")

c1, c2, c3 = st.columns(3)
c1.metric("💰 Jami sotuv", f"{daily['sales_sum'].sum():,.0f}")
c2.metric("↩️ Jami qaytish", f"{daily['return_sum'].sum():,.0f}")
c3.metric("🧠 ML aniqligi", f"{accuracy*100:.2f}%")

# =========================
# 11. ENG ZARARLI MAHSULOTLAR
# =========================
st.subheader("🚨 Eng zararli mahsulotlar")

loss_products = (
    daily.groupby("Номенклатура")["loss_percent"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10,5))
loss_products.plot(kind="bar", ax=ax)
ax.set_ylabel("Zarar %")
ax.set_title("Top 10 zararli mahsulot")
st.pyplot(fig)

st.success("✅ Analiz yakunlandi")
