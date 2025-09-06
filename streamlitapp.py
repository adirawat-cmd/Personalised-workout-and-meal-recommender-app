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

# --- Step 3: Fetch Fitbit Data + Get Recommendations ---
if user_id:
    st.success(f"Using user_id: {user_id}")

    if st.button("📊 Fetch today’s data & Recommend"):
        try:
            # Call FastAPI to fetch Fitbit data
            r = requests.get(urljoin(API_BASE, f"/fetch/daily/{user_id}"))
            r.raise_for_status()
            payload = r.json()

            # Extract inputs for ML model
            steps = 0
            calories_burned = 0
            sleep_hours = 0
            resting_hr = 0
            calories_consumed = 0
            carbs = 0
            protein = 0
            fat = 0

            # Activities
            try:
                activities = payload.get("activities", {}).get("summary", {})
                steps = activities.get("steps", 0)
                calories_burned = activities.get("caloriesOut", 0)
            except Exception:
                pass

            # Sleep
            sleep = payload.get("sleep", {})
            if sleep and sleep.get("summary"):
                mins = sleep["summary"].get("totalMinutesAsleep", 0)
                sleep_hours = round(mins / 60.0, 2)

            # Heart
            hr = payload.get("hr", {})
            try:
                resting_hr = hr.get("activities-heart", [])[0].get("value", {}).get("restingHeartRate", 0)
            except Exception:
                resting_hr = 0

            # Food
            foods = payload.get("foods", {})
            if foods and foods.get("summary"):
                calories_consumed = foods["summary"].get("calories", 0)
                carbs = foods["summary"].get("carbs", 0) or 0
                protein = foods["summary"].get("protein", 0) or 0
                fat = foods["summary"].get("fat", 0) or 0

            row = {
                "steps": steps,
                "calories_burned": calories_burned,
                "resting_hr": resting_hr,
                "sleep_hours": sleep_hours,
                "calories_consumed": calories_consumed,
                "carbs_g": carbs,
                "protein_g": protein,
                "fat_g": fat
            }

            st.write("### Raw extracted inputs")
            st.json(row)

            # Call FastAPI /recommend
            rec_r = requests.post(urljoin(API_BASE, "/recommend"), json={"row": row})
            if rec_r.ok:
                rec = rec_r.json()
                st.write("## 🏋️ Recommendations")
                st.metric("Workout", rec["workout"])
                st.write("🍱 Meal suggestion:", rec["meal"])
            else:
                st.error(f"Recommend failed: {rec_r.text}")

        except Exception as e:
            st.error(f"Fetch failed: {e}")
