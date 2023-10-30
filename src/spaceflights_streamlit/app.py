import pandas as pd
import streamlit as st
from kedro_boot.booter import boot_session

kedro_boot_session = boot_session()

st.title("Spaceflights shuttle price prediction")

with st.container():
    with st.form("my_form"):
        st.write("Shuttle Price Prediction")
        shuttle = {
            "engines": st.slider("engines", 0, 12, 2),
            "passenger_capacity": st.slider("passenger_capacity", 1, 20, 5),
            "crew": st.slider("crew", 0, 20, 5),
            "d_check_complete": st.checkbox("d_check_complete"),
            "moon_clearance_complete": st.checkbox("moon_clearance_complete"),
            "iata_approved": st.checkbox("iata_approved"),
            "company_rating": st.slider("company_rating", 0.0, 1.0, 0.95),
            "review_scores_rating": st.slider("review_score_rating", 20, 100, 88),
        }

        submitted = st.form_submit_button("Predict Price")

if submitted:
    shuttle_df = pd.DataFrame(shuttle, index=[0])
    shuttle_price = kedro_boot_session.run(
        name="inference", inputs={"features_store": shuttle_df}
    )
    shuttle_price = shuttle_price.tolist()
    formated_shuttle_price = round(shuttle_price[0], 3)

    st.success(f"The predicted shuttle price is {formated_shuttle_price} $")
