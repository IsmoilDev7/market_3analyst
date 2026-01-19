import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config("Dekabr 2025 — Real Foyda Analizi", layout="wide")
st.title("💰 Mahsulotlar bo‘yicha REAL foyda & ML tavsiya")

# =====================================================
# FILE UPLOAD
# =====================================================
orders_file = st.file_uploader("📥 Zakazlar (Excel)", type=["xlsx"])
sales_file  = st.file_uploader("📥 Sotuv / Qaytish (Excel)", type=["xlsx"])

if not orders_file or not sales_file:
    st.stop()

orders = pd.read_excel(orders_file)
sales  = pd.read_excel(sales_file)

# =====================================================
# COLUMNS
# =====================================================
orders = orders[[
    "Период", "Номенклатура", "Количество", "Сумма"
]]

sales = sales[[
    "Период", "Номенклатура",
    "Продажная сумма",
    "Себестоимость сумма",
    "Возврат сумма"
]]

# =====================================================
# TYPE FIX
# =====================================================
for df in [orders, sales]:
    df["Период"] = pd.to_datetime(df["Период"], errors="coerce")

orders["Количество"] = orders["Количество"].astype(float)
orders["Сумма"] = orders["Сумма"].astype(str).str.replace(",", "").astype(float)

for col in ["Продажная сумма", "Себестоимость сумма", "Возврат сумма"]:
    sales[col] = sales[col].astype(str).str.replace(",", "").astype(float)

# =====================================================
# DATE FILTER — DEKABR 2025
# =====================================================
date_from = pd.to_datetime("2025-12-01")
date_to   = pd.to_datetime("2025-12-31")

orders = orders[(orders["Период"] >= date_from) & (orders["Период"] <= date_to)]
sales  = sales[(sales["Период"] >= date_from) & (sales["Период"] <= date_to)]

# =====================================================
# AGGREGATION
# =====================================================
orders_agg = orders.groupby("Номенклатура", as_index=False).agg(
    sold_qty=("Количество", "sum"),
    sold_sum=("Сумма", "sum")
)

sales_agg = sales.groupby("Номенклатура", as_index=False).agg(
    cost_sum=("Себестоимость сумма", "sum"),
    return_sum=("Возврат сумма", "sum")
)

df = orders_agg.merge(sales_agg, on="Номенклатура", how="left").fillna(0)

# =====================================================
# REAL PROFIT
# =====================================================
df["real_profit"] = df["sold_sum"] - df["cost_sum"] - df["return_sum"]
df["profit_percent"] = (df["real_profit"] / df["sold_sum"] * 100).clip(-100,100)

df["status"] = np.where(
    df["real_profit"] < 0,
    "❌ ZARAR",
    "✅ FOYDA"
)

# =====================================================
# ML MODEL
# =====================================================
X = df[["sold_qty", "sold_sum", "cost_sum", "return_sum"]]
y = df["profit_percent"]

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    random_state=42
)
model.fit(X, y)

df["ml_profit_forecast"] = model.predict(X).clip(-100,100)

df["ml_recommendation"] = np.where(
    df["ml_profit_forecast"] < 0,
    "❌ To‘xtatish kerak",
    np.where(df["ml_profit_forecast"] < 10,
             "⚠️ Kam hajmda ishlash",
             "✅ Ko‘paytirish mumkin")
)

# =====================================================
# OUTPUT
# =====================================================
st.subheader("📦 Mahsulot bo‘yicha REAL foyda")
st.dataframe(
    df.sort_values("profit_percent"),
    use_container_width=True
)

# =====================================================
# VISUALS
# =====================================================
st.subheader("📊 Foyda % diagramma")
fig, ax = plt.subplots(figsize=(10,5))
df.set_index("Номенклатура")["profit_percent"].plot(kind="bar", ax=ax)
ax.set_ylabel("Foyda %")
st.pyplot(fig)

st.success("""
✅ REAL foyda hisoblandi  
✅ Себестоимость hisobga olindi  
✅ ML tavsiyalar tayyor  
""")
