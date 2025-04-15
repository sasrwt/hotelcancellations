import streamlit as st
import numpy as np
import os
import gdown
import pickle

model_path = "tunedRF.pkl"
gdrive_file_id = "1Sw8QdQ4v-S7uy8F7u4xCbYZm7Nl9pqlc"

# Download model from Google Drive if not present
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={gdrive_file_id}", model_path, quiet=False)

# Load the model
model = pickle.load(open(model_path, 'rb'))







def main():
    st.title("🌴 Predicting Cancellations at Hotel Carolina")
    st.markdown("#### 🎶 Inspired by *Hotel California* — but with a twist: data-driven decisions meet dreamy destinations.")

    with open("hotelcal.gif", "rb") as gif_file:
  
        gif_bytes = gif_file.read()
        st.image(gif_bytes)

    # --- Guest & Booking Info ---
    no_of_adults = st.selectbox('Number of adults', options=list(range(0, 7)), index=2)
    no_of_children = st.selectbox('Number of children', options=list(range(0, 9)), index=0)
    no_of_weekend_nights = st.selectbox('Number of weekend nights', options=list(range(0, 3)))
    no_of_week_nights = st.number_input('Number of week nights', step=1, min_value=0)

    meal_plan_name = st.selectbox('Meal Plan', options=['Not Selected', 'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3'])
    meal_plan_map = {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3}
    meal_plan = meal_plan_map[meal_plan_name]

    st.markdown("---")
    required_car_parking_space = 1 if st.checkbox('Car parking space required') else 0
    st.markdown("---")

    room_type_reserved = st.selectbox('Type of room reserved (0–6)', options=list(range(0, 7)))
    lead_time = st.number_input(
        'Number of days before the arrival date the booking was made.',
        step=1, min_value=0
    )

    arrival_year = st.selectbox('Year of arrival', options=[2017, 2018, 2019])
    arrival_month = st.selectbox('Month of arrival', options=list(range(1, 13)))
    arrival_date = st.selectbox('Date of the month for arrival', options=list(range(1, 32)))

    market_segment_label = st.selectbox('Booking method', options=[
        'Aviation', 'Complementary', 'Corporate', 'Offline', 'Online'
    ])

    # Manually one-hot encode market_segment_type (drop_first=True: Aviation dropped)
    market_segment_dummies = [0, 0, 0, 0]  # Order: Complementary, Corporate, Offline, Online
    if market_segment_label == 'Complementary':
        market_segment_dummies[0] = 1
    elif market_segment_label == 'Corporate':
        market_segment_dummies[1] = 1
    elif market_segment_label == 'Offline':
        market_segment_dummies[2] = 1
    elif market_segment_label == 'Online':
        market_segment_dummies[3] = 1
    # Else: Aviation → all zeros (reference category)

    repeated_guest = st.selectbox('Has the guest stayed before?', options=['No', 'Yes'])
    repeated_guest = 1 if repeated_guest == 'Yes' else 0

    no_of_previous_cancellations = st.number_input('Number of previous cancellations', step=1, min_value=0)
    no_of_previous_bookings_not_cancelled = st.number_input('Number of previous bookings not canceled', step=1, min_value=0)

    avg_price_per_room = st.number_input(
        'Average price per day of the booking ($1 – $499)',
        step=1, min_value=1, max_value=499
    )
    no_of_special_requests = st.number_input('Number of special requests', step=1, min_value=0)

    # Engineered Features
    no_of_individuals = no_of_adults + no_of_children
    no_of_days_booked = no_of_weekend_nights + no_of_week_nights

    # Final feature list for model
    user_input = [
        no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
        meal_plan, required_car_parking_space, room_type_reserved, lead_time,
        arrival_year, arrival_month, arrival_date,
        repeated_guest, no_of_previous_cancellations,
        no_of_previous_bookings_not_cancelled, avg_price_per_room,
        no_of_special_requests, no_of_individuals, no_of_days_booked
    ] + market_segment_dummies

    if st.button('Predict'):
        prediction = model.predict(np.array([user_input]))
        output = prediction[0]
        if output == 1:
            st.success("✅ The booking will not be canceled.")
        else:
            st.error("❌ The booking will be canceled.")

    st.markdown("---")
    st.markdown("<small>Created by Dr. T with ❤️</small>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
