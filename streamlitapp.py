# # # # streamlit_app.py
# # # import streamlit as st
# # # import requests
# # # import time
# # # import pandas as pd
# # # from urllib.parse import urljoin

# # # API_BASE = "http://127.0.0.1:8000"  # your FastAPI URL

# # # st.set_page_config(page_title="Fitbit Recommender", layout="centered")

# # # st.title("Personalized Workout & Nutrition — Connect Fitbit")

# # # col1, col2 = st.columns([1,2])

# # # with col1:
# # #     if st.button("Connect Fitbit"):
# # #         # open login endpoint in new tab - user completes login there
# # #         login_url = urljoin(API_BASE, "/login")
# # #         st.markdown(f"[Click here to authorize Fitbit]({login_url})", unsafe_allow_html=True)
# # #         st.info("After authorizing, you'll see a short 'connected' page with your Fitbit user id. Copy that id into the box below.")

# # # with col2:
# # #     user_id = st.text_input("Enter your Fitbit user_id (from the connected page)", value="", help="Example: C12345; shown when Fitbit redirects back in your browser.")
# # #     if st.button("List connected users"):
# # #         r = requests.get(urljoin(API_BASE, "/connected"))
# # #         if r.ok:
# # #             st.write("Connected user_ids:", r.json())
# # #         else:
# # #             st.error("Failed to fetch connected users")

# # # if user_id:
# # #     st.success(f"Selected user_id: {user_id}")

# # #     if st.button("Fetch today data & Recommend"):
# # #         try:
# # #             r = requests.get(urljoin(API_BASE, f"/fetch/daily/{user_id}"))
# # #             r.raise_for_status()
# # #             payload = r.json()
# # #             # extract features safely from payload (adapt to what your model needs)
# # #             # Example extraction — tweak to match fitbit response & your model features:
# # #             steps = 0
# # #             calories_burned = 0
# # #             sleep_hours = 0
# # #             resting_hr = None
# # #             calories_consumed = 0
# # #             carbs = 0; protein = 0; fat = 0

# # #             # Activities steps
# # #             activities = payload.get("activities", {})
# # #             try:
# # #                 steps = int(activities.get("activities-steps", [{"value": 0}])[0]["value"])
# # #                 calories_burned = int(activities.get("activities-calories", [{"value": 0}])[0]["value"])
# # #             except Exception:
# # #                 # fallback: try aggregated summary
# # #                 pass

# # #             # Sleep
# # #             sleep = payload.get("sleep", {})
# # #             if sleep and sleep.get("summary"):
# # #                 mins = sleep["summary"].get("totalMinutesAsleep", 0)
# # #                 sleep_hours = round(mins / 60.0, 2)

# # #             # Heart
# # #             hr = payload.get("hr", {})
# # #             try:
# # #                 resting_hr = hr.get("activities-heart", [])[0].get("value", {}).get("restingHeartRate")
# # #             except Exception:
# # #                 resting_hr = None

# # #             # Food
# # #             foods = payload.get("foods", {})
# # #             if foods and foods.get("summary"):
# # #                 calories_consumed = foods["summary"].get("calories", 0)
# # #                 carbs = foods["summary"].get("carbs", 0) or 0
# # #                 protein = foods["summary"].get("protein", 0) or 0
# # #                 fat = foods["summary"].get("fat", 0) or 0

# # #             row = {
# # #                 "steps": steps,
# # #                 "calories_burned": calories_burned,
# # #                 "resting_hr": resting_hr or 0,
# # #                 "sleep_hours": sleep_hours,
# # #                 "calories_consumed": calories_consumed,
# # #                 "carbs_g": carbs,
# # #                 "protein_g": protein,
# # #                 "fat_g": fat
# # #             }

# # #             st.write("### Raw extracted inputs")
# # #             st.json(row)

# # #             # call local recommend function (we will call our recommend.py through an endpoint OR import it).
# # #             # For simplicity we call a FastAPI endpoint in backend that wraps recommend.py
# # #             rec_r = requests.post(urljoin(API_BASE, "/recommend"), json={"row": row})
# # #             if rec_r.ok:
# # #                 rec = rec_r.json()
# # #                 st.write("## Recommendations")
# # #                 st.metric("Workout", rec["workout"])
# # #                 st.write("Meal suggestion:", rec["meal"])
# # #             else:
# # #                 st.error(f"Recommend failed: {rec_r.text}")

# # #         except Exception as e:
# # #             st.error(f"Fetch failed: {e}")

# # # if "code" in st.query_params:
# # #     auth_code = st.query_params["code"]
# # #     tokens = get_tokens(auth_code)  # from earlier code

# # #     if "access_token" in tokens:
# # #         st.success("✅ Connected to Fitbit!")

# # #         # Save tokens locally for this user
# # #         with open("fitbit_tokens.json", "w") as f:
# # #             import json
# # #             json.dump(tokens, f)

# # #         st.json(tokens)
# # #     else:
# # #         st.error("Failed to connect Fitbit")
# # #         st.json(tokens)


# # # def get_daily_summary(access_token):
# # #     headers = {"Authorization": f"Bearer {access_token}"}
# # #     url = "https://api.fitbit.com/1/user/-/activities/date/today.json"
# # #     response = requests.get(url, headers=headers)
# # #     return response.json()

# # # streamlit_app.py
# # import streamlit as st
# # import requests
# # from urllib.parse import urljoin

# # API_BASE = "http://127.0.0.1:8000"  # FastAPI backend

# # st.set_page_config(page_title="Fitbit Recommender", layout="centered")
# # st.title("Personalized Workout & Nutrition — Connect Fitbit")

# # # --- Step 1: Connect to Fitbit ---
# # if st.button("Connect Fitbit"):
# #     login_url = urljoin(API_BASE, "/login")
# #     st.markdown(f"[Click here to authorize Fitbit]({login_url})", unsafe_allow_html=True)
# #     st.info("After authorizing, you'll see a 'connected' page with your Fitbit user_id. Copy it below.")

# # # --- Step 2: User selects their Fitbit ID ---
# # user_id = st.text_input("Enter your Fitbit user_id", value="", help="Shown after Fitbit login.")

# # if st.button("List connected users"):
# #     r = requests.get(urljoin(API_BASE, "/connected"))
# #     if r.ok:
# #         st.write("Connected user_ids:", r.json())
# #     else:
# #         st.error("Failed to fetch connected users")

# # if user_id:
# #     st.success(f"Selected user_id: {user_id}")

# #     # --- Step 3: Fetch Fitbit data + Recommendations ---
# #     if st.button("Fetch today’s data & Recommend"):
# #         try:
# #             # 1. Get Fitbit data from backend
# #             r = requests.get(urljoin(API_BASE, f"/fetch/daily/{user_id}"))
# #             r.raise_for_status()
# #             payload = r.json()

# #             # Show raw Fitbit response
# #             st.write("### Raw Fitbit Data")
# #             st.json(payload)

# #             # 2. Call backend /recommend endpoint
# #             rec_r = requests.post(urljoin(API_BASE, "/recommend"), json={"row": payload})
# #             if rec_r.ok:
# #                 rec = rec_r.json()
# #                 st.write("## Recommendations")
# #                 st.metric("💪 Workout", rec["workout"])
# #                 st.write("🍽️ Meal suggestion:", rec["meal"])
# #             else:
# #                 st.error(f"Recommend failed: {rec_r.text}")

# #         except Exception as e:
# #             st.error(f"Fetch failed: {e}")

# # streamlit_app.py
# import streamlit as st
# import requests
# from urllib.parse import urljoin

# API_BASE = "http://127.0.0.1:8000"  # FastAPI backend

# st.set_page_config(page_title="Fitbit Recommender", layout="centered")
# st.title("💪 Personalized Workout & 🍽️ Nutrition Recommender")

# # --- Step 1: Login with Fitbit ---
# if st.button("🔗 Connect Fitbit"):
#     login_url = urljoin(API_BASE, "/login")
#     st.markdown(f"[Click here to authorize Fitbit]({login_url})", unsafe_allow_html=True)
#     st.info("After authorizing, you'll see a page with your Fitbit user_id. Copy-paste it below.")

# # --- Step 2: Select user_id ---
# user_id = st.text_input(
#     "Enter your Fitbit user_id (from the connected page):",
#     value="",
#     help="Example: 123ABC (displayed after login)"
# )

# if st.button("👥 Show Connected Users"):
#     try:
#         r = requests.get(urljoin(API_BASE, "/connected"))
#         if r.ok:
#             st.write("Connected users:", r.json())
#         else:
#             st.error(f"Error: {r.text}")
#     except Exception as e:
#         st.error(f"Failed to connect API: {e}")

# # --- Step 3: Fetch Fitbit Data + Get Recommendations ---
# if user_id:
#     st.success(f"Using user_id: {user_id}")

#     if st.button("📊 Fetch today’s data & Recommend"):
#         try:
#             # Call FastAPI to fetch Fitbit data
#             r = requests.get(urljoin(API_BASE, f"/fetch/daily/{user_id}"))
#             r.raise_for_status()
#             payload = r.json()

#             # Extract inputs for ML model
#             steps = 0
#             calories_burned = 0
#             sleep_hours = 0
#             resting_hr = 0
#             calories_consumed = 0
#             carbs = 0
#             protein = 0
#             fat = 0

#             # Activities
#             try:
#                 activities = payload.get("activities", {}).get("summary", {})
#                 steps = activities.get("steps", 0)
#                 calories_burned = activities.get("caloriesOut", 0)
#             except Exception:
#                 pass

#             # Sleep
#             sleep = payload.get("sleep", {})
#             if sleep and sleep.get("summary"):
#                 mins = sleep["summary"].get("totalMinutesAsleep", 0)
#                 sleep_hours = round(mins / 60.0, 2)

#             # Heart
#             hr = payload.get("hr", {})
#             try:
#                 resting_hr = hr.get("activities-heart", [])[0].get("value", {}).get("restingHeartRate", 0)
#             except Exception:
#                 resting_hr = 0

#             # Food
#             foods = payload.get("foods", {})
#             if foods and foods.get("summary"):
#                 calories_consumed = foods["summary"].get("calories", 0)
#                 carbs = foods["summary"].get("carbs", 0) or 0
#                 protein = foods["summary"].get("protein", 0) or 0
#                 fat = foods["summary"].get("fat", 0) or 0

#             row = {
#                 "steps": steps,
#                 "calories_burned": calories_burned,
#                 "resting_hr": resting_hr,
#                 "sleep_hours": sleep_hours,
#                 "calories_consumed": calories_consumed,
#                 "carbs_g": carbs,
#                 "protein_g": protein,
#                 "fat_g": fat
#             }

#             st.write("### Raw extracted inputs")
#             st.json(row)

#             # Call FastAPI /recommend
#             rec_r = requests.post(urljoin(API_BASE, "/recommend"), json={"row": row})
#             if rec_r.ok:
#                 rec = rec_r.json()
#                 st.write("## 🏋️ Recommendations")
#                 st.metric("Workout", rec["workout"])
#                 st.write("🍱 Meal suggestion:", rec["meal"])
#             else:
#                 st.error(f"Recommend failed: {rec_r.text}")

#         except Exception as e:
#             st.error(f"Fetch failed: {e}")

import streamlit as st
import requests
from urllib.parse import urljoin

API_BASE = "http://127.0.0.1:8000"  # FastAPI backend

st.set_page_config(page_title="Fitbit Recommender", layout="centered")
st.title("💪 Personalized Workout & 🍽️ Nutrition Recommender")

# --- Step 1: Login with Fitbit ---
if st.button("🔗 Connect Fitbit"):
    login_url = urljoin(API_BASE, "/login")
    st.markdown(f"[Click here to authorize Fitbit]({login_url})", unsafe_allow_html=True)
    st.info("After authorizing, you'll see a page with your Fitbit user_id. Copy-paste it below.")

# --- Step 2: Select user_id ---
user_id = st.text_input(
    "Enter your Fitbit user_id (from the connected page):",
    value="",
    help="Example: 123ABC (displayed after login)"
)

if st.button("👥 Show Connected Users"):
    try:
        r = requests.get(urljoin(API_BASE, "/connected"))
        if r.ok:
            st.write("Connected users:", r.json())
        else:
            st.error(f"Error: {r.text}")
    except Exception as e:
        st.error(f"Failed to connect API: {e}")

# --- Step 3: Fetch Fitbit Data + Recommendations ---
if user_id:
    st.success(f"Using user_id: {user_id}")

    if st.button("📊 Fetch Today’s Data & Recommend"):
        try:
            # Call backend to get both data + recommendation
            r = requests.get(urljoin(API_BASE, f"/recommend/daily/{user_id}"))
            r.raise_for_status()
            payload = r.json()

            features = payload["features"]
            rec = payload["recommendation"]

            # --- Show daily summary nicely ---
            st.subheader("📊 Your Daily Summary")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Steps", features["steps"])
            col2.metric("Calories Burned", features["calories_burned"])
            col3.metric("Sleep (hrs)", round(features["sleep_hours"], 1))
            col4.metric("Resting HR", features["resting_hr"])

            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Calories In", features["calories_consumed"])
            col6.metric("Carbs (g)", features["carbs_g"])
            col7.metric("Protein (g)", features["protein_g"])
            col8.metric("Fat (g)", features["fat_g"])

            # --- Show ML recommendation ---
            st.subheader("🤖 AI Recommendation")
            st.success(f"🏋️ Workout: **{rec['workout']}**")
            st.info(f"🍱 Meal: {rec['meal']}")

        except Exception as e:
            st.error(f"Fetch failed: {e}")
