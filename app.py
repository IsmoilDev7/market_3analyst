import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Market Analyst", layout="wide")

# =========================
# 1. DATA YUKLASH
# =========================
st.title("📊 Mahsulotlar bo‘yicha zarar / foyda analizi")

uploaded_file = st.file_uploader("CSV faylni yuklang", type=["csv"])

if uploaded_file:
    sales = pd.read_csv(uploaded_file)

    sales.columns = sales.columns.str.strip().str.lower()

    def find_col(names):
        for c in sales.columns:
            if c in names:
                return c
        return None

    date_col = find_col(["период", "дата", "date"])
    prod_col = find_col(["номенклатура", "товар", "product"])
    qty_col  = find_col(["количество", "qty"])
    ret_col  = find_col(["возврат количество", "возврат", "return"])
    sum_col  = find_col(["сумма", "sales_sum"])

    if None in [date_col, prod_col, qty_col, ret_col]:
        st.error("❌ Kerakli ustunlar topilmadi")
        st.stop()

    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce")
    sales = sales.dropna(subset=[date_col])

    # =========================
    # 2. SANA FILTRI
    # =========================
    col1, col2 = st.columns(2)
    date_from = col1.date_input("Boshlanish sana", sales[date_col].min())
    date_to   = col2.date_input("Tugash sana", sales[date_col].max())

    sales = sales[
        (sales[date_col] >= pd.to_datetime(date_from)) &
        (sales[date_col] <= pd.to_datetime(date_to))
    ]

    sales["day"] = sales[date_col].dt.date

    # =========================
    # 3. KUNLIK AGREGATSIYA
    # =========================
    daily = sales.groupby(["day", prod_col], as_index=False).agg(
        sold_qty=(qty_col, "sum"),
        return_qty=(ret_col, "sum")
    )

    daily["loss_percent"] = np.where(
        daily["sold_qty"] > 0,
        (daily["return_qty"] / daily["sold_qty"]) * 100,
        0
    ).clip(0, 100)

    # =========================
    # 4. ZARAR / FOYDA LABEL
    # =========================
    daily["label"] = np.where(daily["loss_percent"] > 20, 1, 0)
    # 1 = ZARAR, 0 = FOYDA

    # =========================
    # 5. ML MODEL
    # =========================
    X = daily[["sold_qty", "return_qty", "loss_percent"]]
    y = daily["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    # =========================
    # 6. PREDICTION
    # =========================
    daily["prediction"] = model.predict(X)
    daily["prediction_text"] = daily["prediction"].map({
        1: "❌ ZARAR keltiradi",
        0: "✅ FOYDA keltiradi"
    })

    # =========================
    # 7. DASHBOARD
    # =========================
    st.subheader("📌 Model aniqligi")
    st.success(f"Accuracy: {round(acc * 100, 2)}%")

    st.subheader("📋 Mahsulotlar bo‘yicha natija")
    st.dataframe(daily)

    # =========================
    # 8. GRAFIK
    # =========================
    st.subheader("📈 Eng zararli mahsulotlar")

    top_loss = (
        daily.groupby(prod_col)["loss_percent"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    top_loss.plot(kind="bar", ax=ax)
    ax.set_ylabel("Zarar %")
    st.pyplot(fig)

else:
    st.info("👆 CSV fayl yuklang")
