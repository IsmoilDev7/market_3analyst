import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config("Dekabr 2025 Analitika", layout="wide")
st.title("📊 Dekabr 2025 — Mahsulotlar bo‘yicha chuqur analiz")

# =====================================================
# 1. EXCEL YUKLASH
# =====================================================
orders_file = st.file_uploader("📥 Zakazlar Excel", type=["xlsx"])
returns_file = st.file_uploader("📥 Sotuv / Qaytish Excel", type=["xlsx"])

if not orders_file or not returns_file:
    st.stop()

orders = pd.read_excel(orders_file)
returns = pd.read_excel(returns_file)

# =====================================================
# 2. KERAKLI USTUNLAR
# =====================================================
orders = orders[[
    "Период", "Номенклатура", "Количество", "Сумма"
]]

returns = returns[[
    "Период", "Номенклатура", "Количество",
    "Продажная сумма", "Возврат сумма"
]]

# =====================================================
# 3. TYPE FIX
# =====================================================
for df in [orders, returns]:
    df["Период"] = pd.to_datetime(df["Период"], errors="coerce")

orders["Количество"] = orders["Количество"].astype(float)
orders["Сумма"] = orders["Сумма"].astype(str).str.replace(",", "").astype(float)

returns["Количество"] = returns["Количество"].astype(float)
returns["Возврат сумма"] = returns["Возврат сумма"].astype(str).str.replace(",", "").astype(float)

# =====================================================
# 4. 30 KUNLIK DEKABR FILTER
# =====================================================
date_from = st.date_input("📅 Sana boshlanishi", pd.to_datetime("2025-12-01"))
date_to   = st.date_input("📅 Sana oxiri", pd.to_datetime("2025-12-31"))

orders = orders[(orders["Период"] >= pd.to_datetime(date_from)) &
                (orders["Период"] <= pd.to_datetime(date_to))]

returns = returns[(returns["Период"] >= pd.to_datetime(date_from)) &
                  (returns["Период"] <= pd.to_datetime(date_to))]

orders["day"] = orders["Период"].dt.date
returns["day"] = returns["Период"].dt.date

# =====================================================
# 5. KUNLIK + MAHSULOT ANALIZI
# =====================================================
daily_orders = orders.groupby(["day", "Номенклатура"], as_index=False).agg(
    sold_qty=("Количество", "sum"),
    sold_sum=("Сумма", "sum")
)

daily_returns = returns.groupby(["day", "Номенклатура"], as_index=False).agg(
    return_sum=("Возврат сумма", "sum")
)

daily = pd.merge(
    daily_orders, daily_returns,
    on=["day", "Номенклатура"], how="left"
).fillna(0)

# =====================================================
# 6. FOYDA / ZARAR %
# =====================================================
daily["loss_percent"] = (daily["return_sum"] / daily["sold_sum"] * 100).clip(0,100)
daily["profit_percent"] = 100 - daily["loss_percent"]

daily["status"] = np.where(
    daily["loss_percent"] > 20,
    "❌ ZARARLI",
    "✅ FOYDALI"
)

# =====================================================
# 7. MAHSULOT BO‘YICHA YAKUNIY ANALIZ
# =====================================================
product_summary = daily.groupby("Номенклатура", as_index=False).agg(
    sold_sum=("sold_sum", "sum"),
    return_sum=("return_sum", "sum"),
    avg_loss_percent=("loss_percent", "mean"),
    avg_profit_percent=("profit_percent", "mean")
)

product_summary["status"] = np.where(
    product_summary["avg_loss_percent"] > 20,
    "❌ ZARARLI",
    "✅ FOYDALI"
)

# =====================================================
# 8. ML: 100% FOYDA STRATEGIYASI
# =====================================================
X = daily[["sold_qty", "sold_sum"]]
y = daily["loss_percent"]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)
model.fit(X, y)

daily["predicted_loss"] = model.predict(X).clip(0,100)
daily["recommended_profit"] = 100 - daily["predicted_loss"]

# =====================================================
# 9. JADVALLAR
# =====================================================
st.subheader("📦 Har bir mahsulot bo‘yicha yakuniy natija")
st.dataframe(product_summary.sort_values("avg_loss_percent", ascending=False),
             use_container_width=True)

st.subheader("📅 Kunlik (30 kun) batafsil analiz")
st.dataframe(daily.sort_values("loss_percent", ascending=False),
             use_container_width=True)

# =====================================================
# 10. DIAGRAMMALAR
# =====================================================
st.subheader("📊 Eng zararli mahsulotlar (%)")
fig, ax = plt.subplots(figsize=(10,5))
product_summary.set_index("Номенклатура")["avg_loss_percent"].plot(
    kind="bar", ax=ax
)
ax.set_ylabel("Zarar %")
st.pyplot(fig)

st.subheader("📈 Kunlik zarar dinamikasi")
fig2, ax2 = plt.subplots(figsize=(10,5))
daily.groupby("day")["loss_percent"].mean().plot(ax=ax2)
ax2.set_ylabel("O‘rtacha zarar %")
st.pyplot(fig2)

# =====================================================
# 11. XULOSA
# =====================================================
st.success("""
✅ Har bir mahsulotning foyda / zarar foizi hisoblandi  
✅ 30 kunlik kunlik analiz qilindi  
✅ ML orqali zarar ehtimoli va foyda strategiyasi topildi  
""")
