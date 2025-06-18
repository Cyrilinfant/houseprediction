import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    'price': [550000, 720000, 330000, 875000, 450000, 990000, 625000],
    'bedrooms': [3, 4, 2, 5, 3, 6, 4],
    'bathrooms': [2, 3, 1, 4, 2.5, 5, 2.5],
    'sqft_living': [1800, 2400, 1200, 3200, 2000, 4000, 2600],
    'sqft_lot': [5000, 6000, 3000, 7500, 5500, 10000, 7200],
    'floors': [1, 2, 1, 2, 2, 3, 2],
    'waterfront': [0, 1, 0, 1, 0, 1, 0],
    'view': [1, 2, 0, 3, 1, 4, 2],
    'condition': [3, 4, 3, 5, 4, 5, 4],
    'grade': [7, 8, 6, 9, 7, 10, 8],
    'yr_built': [1995, 2005, 1980, 2015, 1998, 2020, 2008],
    'yr_renovated': [0, 2010, 0, 2018, 2005, 0, 2012]
}
df = pd.DataFrame(data)

x = df.drop('price', axis=1)
y = df['price']

model = LinearRegression()
model.fit(x, y)

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", page_icon="🏡")
st.title("🏡 House Price Prediction")
st.write("Enter your house requirements and click **Predict** to get an estimated price.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("🛏 Bedrooms", min_value=0, value=3)
        bathrooms = st.number_input("🛁 Bathrooms", min_value=0, value=2)
        sqft_living = st.number_input("📐 Living Area (sqft)", min_value=0, value=1800)
        sqft_lot = st.number_input("🌳 Lot Area (sqft)", min_value=0, value=5000)
        floors = st.number_input("🏢 Floors", min_value=0, value=1)

    with col2:
        waterfront = st.selectbox("🌊 Waterfront View?", [0, 1])
        view = st.slider("👀 View Rating", 0, 4, 1)
        condition = st.slider("🏚 Condition Rating", 1, 5, 3)
        grade = st.slider("🏗 Grade", 1, 10, 7)
        yr_built = st.number_input("🛠 Year Built", min_value=1800, value=1995)
        yr_renovated = st.number_input("🔧 Year Renovated (0 if never)", min_value=0, value=0)

    submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        input_data = pd.DataFrame([[
            bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront,
            view, condition, grade, yr_built, yr_renovated
        ]], columns=x.columns)

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated House Price: **₹{int(prediction):,}**")
