import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Dekabr 2025 Mahsulot Analizi", layout="wide")
st.title("📊 Dekabr 2025 — Mahsulotlar bo‘yicha foyda / zarar analizi")

# ==================================================
# 1. EXCEL FAYLLAR
# ==================================================
orders_file = st.file_uploader("1️⃣ Zakazlar Excel", type=["xlsx", "xls"])
returns_file = st.file_uploader("2️⃣ Sotuv / Qaytish Excel", type=["xlsx", "xls"])

if not orders_file or not returns_file:
    st.info("Ikkala Excel faylni yuklang")
    st.stop()

orders = pd.read_excel(orders_file)
returns = pd.read_excel(returns_file)

# ==================================================
# 2. USTUNLARNI NORMALIZATSIYA
# ==================================================
orders = orders[[
    "Период",
    "Номенклатура",
    "Контрагент",
    "Количество",
    "Сумма"
]]

returns = returns[[
    "Период",
    "Номенклатура",
    "Контрагент",
    "Количество",
    "Продажная сумма",
    "Возврат сумма"
]]

# ==================================================
# 3. DATA TYPE FIX
# ==================================================
for df in [orders, returns]:
    df["Период"] = pd.to_datetime(df["Период"], errors="coerce")

orders["Количество"] = orders["Количество"].astype(float)
orders["Сумма"] = orders["Сумма"].astype(str).str.replace(",", "").astype(float)

returns["Количество"] = returns["Количество"].astype(float)
returns["Продажная сумма"] = returns["Продажная сумма"].fillna(0)
returns["Возврат сумма"] = returns["Возврат сумма"].astype(str).str.replace(",", "").astype(float)

# ==================================================
# 4. DEKABR 2025 FILTR
# ==================================================
start = pd.to_datetime("2025-12-01")
end   = pd.to_datetime("2025-12-31")

orders = orders[(orders["Период"] >= start) & (orders["Период"] <= end)]
returns = returns[(returns["Период"] >= start) & (returns["Период"] <= end)]

orders["day"] = orders["Период"].dt.date
returns["day"] = returns["Период"].dt.date

# ==================================================
# 5. KUNLIK MAHSULOT ANALIZI
# ==================================================
daily_orders = orders.groupby(
    ["day", "Номенклатура"], as_index=False
).agg(
    order_qty=("Количество", "sum"),
    order_sum=("Сумма", "sum")
)

daily_returns = returns.groupby(
    ["day", "Номенклатура"], as_index=False
).agg(
    return_qty=("Количество", "sum"),
    return_sum=("Возврат сумма", "sum")
)

daily = pd.merge(
    daily_orders,
    daily_returns,
    on=["day", "Номенклатура"],
    how="left"
).fillna(0)

# ==================================================
# 6. ZARAR FOYDA HISOBI
# ==================================================
daily["loss_percent"] = np.where(
    daily["order_sum"] > 0,
    (daily["return_sum"] / daily["order_sum"]) * 100,
    0
).clip(0, 100)

daily["status"] = np.where(
    daily["loss_percent"] > 20,
    "❌ ZARARLI",
    "✅ FOYDALI"
)

# ==================================================
# 7. ML UCHUN LABEL
# ==================================================
daily["label"] = (daily["loss_percent"] > 20).astype(int)

features = [
    "order_qty",
    "order_sum",
    "return_qty",
    "return_sum",
    "loss_percent"
]

X = daily[features]
y = daily["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X_train, y_train)

daily["ML_bashorat"] = model.predict(X)
daily["ML_natija"] = daily["ML_bashorat"].map({
    1: "❌ ZARAR keltiradi",
    0: "✅ FOYDA keltiradi"
})

accuracy = accuracy_score(y_test, model.predict(X_test))

# ==================================================
# 8. JADVAL
# ==================================================
st.subheader("📋 Kunlik mahsulotlar natijasi")
st.dataframe(
    daily.sort_values(["loss_percent"], ascending=False),
    use_container_width=True
)

# ==================================================
# 9. ENG ZARARLI MAHSULOTLAR
# ==================================================
st.subheader("🚨 Dekabr oyidagi eng zararli mahsulotlar")

loss_products = (
    daily.groupby("Номенклатура")["loss_percent"]
    .mean()
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10,5))
loss_products.plot(kind="bar", ax=ax)
ax.set_ylabel("Zarar %")
ax.set_title("Mahsulotlar bo‘yicha o‘rtacha zarar")
st.pyplot(fig)

# ==================================================
# 10. KPI
# ==================================================
st.subheader("📌 Umumiy ko‘rsatkichlar")

c1, c2, c3 = st.columns(3)
c1.metric("💰 Jami sotuv", f"{daily['order_sum'].sum():,.0f}")
c2.metric("↩️ Jami qaytish", f"{daily['return_sum'].sum():,.0f}")
c3.metric("🧠 ML aniqligi", f"{accuracy*100:.2f}%")

st.success("✅ Analiz to‘liq yakunlandi")
