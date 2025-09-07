# # api.py
# import os
# import json
# from pathlib import Path
# from datetime import date, datetime
# from typing import Dict
# import requests
# from requests.auth import HTTPBasicAuth
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from urllib.parse import quote

# from dotenv import load_dotenv
# from recommend import recommend as model_recommend

# load_dotenv()

# CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
# CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
# REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8000/callback")
# TOKEN_URL = "https://api.fitbit.com/oauth2/token"
# AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"

# # Where we persist tokens locally for demo. Use a DB in production.
# TOKENS_FILE = Path("tokens.json")

# app = FastAPI(title="Fitbit OAuth Demo")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501"],  # Streamlit frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# SCOPES = ["activity", "heartrate", "sleep", "nutrition"]


# # ---------------- Token storage helpers ----------------
# def load_tokens() -> Dict:
#     if TOKENS_FILE.exists():
#         return json.loads(TOKENS_FILE.read_text())
#     return {}

# def save_tokens(all_tokens: Dict):
#     TOKENS_FILE.write_text(json.dumps(all_tokens, indent=2))

# def store_user_tokens(fitbit_user_id: str, access_token: str, refresh_token: str):
#     tokens = load_tokens()
#     tokens[fitbit_user_id] = {
#         "access_token": access_token,
#         "refresh_token": refresh_token
#     }
#     save_tokens(tokens)

# def get_user_tokens(fitbit_user_id: str):
#     return load_tokens().get(fitbit_user_id)

# def refresh_access_token_for_user(fitbit_user_id: str):
#     tokens = get_user_tokens(fitbit_user_id)
#     if not tokens:
#         raise HTTPException(status_code=404, detail="User not connected")

#     refresh_token = tokens["refresh_token"]
#     resp = requests.post(
#         TOKEN_URL,
#         data={
#             "grant_type": "refresh_token",
#             "refresh_token": refresh_token,
#             "client_id": CLIENT_ID,
#         },
#         auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
#         headers={"Content-Type": "application/x-www-form-urlencoded"},
#     )
#     if resp.status_code != 200:
#         raise HTTPException(status_code=400, detail=f"Refresh failed: {resp.text}")

#     data = resp.json()
#     # store new tokens
#     store_user_tokens(data["user_id"], data["access_token"], data["refresh_token"])
#     return data


# # ---------------- OAuth routes ----------------
# @app.get("/login")
# def login():
#     """Redirect user to Fitbit auth page."""
#     scope = "%20".join(SCOPES)
#     encoded_redirect = quote(REDIRECT_URI, safe="")
#     url = (
#         f"{AUTHORIZE_URL}?response_type=code&client_id={CLIENT_ID}"
#         f"&redirect_uri={encoded_redirect}&scope={scope}"
#     )
#     return RedirectResponse(url)

# @app.get("/callback")
# def callback(request: Request, code: str = None, error: str = None):
#     """Fitbit will redirect here with ?code=..."""
#     if error:
#         return HTMLResponse(f"<h3>Auth failed: {error}</h3>")

#     if code is None:
#         return HTMLResponse("<h3>No code provided by Fitbit</h3>")

#     # Exchange authorization code for tokens
#     resp = requests.post(
#         TOKEN_URL,
#         data={
#             "client_id": CLIENT_ID,
#             "grant_type": "authorization_code",
#             "redirect_uri": REDIRECT_URI,
#             "code": code
#         },
#         auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
#         headers={"Content-Type": "application/x-www-form-urlencoded"},
#     )

#     if resp.status_code != 200:
#         return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{resp.text}</pre>", status_code=400)

#     data = resp.json()
#     user_id = data.get("user_id") or data.get("user", {}).get("encodedId") or "unknown"
#     store_user_tokens(user_id, data["access_token"], data["refresh_token"])

#     html = f"""
#     <h3>Fitbit connected ✅</h3>
#     <p>User id: <b>{user_id}</b></p>
#     <p>Close this tab and return to the Streamlit app. Use this user id to fetch your data.</p>
#     """
#     return HTMLResponse(html)

# @app.get("/connected")
# def connected_users():
#     """Return list of connected Fitbit user_ids."""
#     tokens = load_tokens()
#     return JSONResponse(list(tokens.keys()))


# # ---------------- Fetch daily data ----------------
# def fitbit_get(path: str, access_token: str):
#     url = f"https://api.fitbit.com{path}"
#     headers = {"Authorization": f"Bearer {access_token}"}
#     r = requests.get(url, headers=headers)
#     return r


# @app.get("/fetch/daily/{user_id}")
# def fetch_daily(user_id: str):
#     """Fetch aggregated daily summary (steps, calories, sleep, food, HR)."""
#     tokens = get_user_tokens(user_id)
#     if not tokens:
#         raise HTTPException(status_code=404, detail="User not connected")

#     access_token = tokens["access_token"]
#     today = date.today().strftime("%Y-%m-%d")

#     # Activities
#     r = fitbit_get(f"/1/user/-/activities/date/{today}.json", access_token)
#     if r.status_code == 401:
#         refresh_access_token_for_user(user_id)
#         tokens = get_user_tokens(user_id)
#         access_token = tokens["access_token"]
#         r = fitbit_get(f"/1/user/-/activities/date/{today}.json", access_token)

#     if r.status_code != 200:
#         raise HTTPException(status_code=r.status_code, detail=r.text)
#     activities = r.json()

#     # Sleep
#     r_sleep = fitbit_get(f"/1.2/user/-/sleep/date/{today}.json", access_token)
#     sleep_data = r_sleep.json() if r_sleep.status_code == 200 else {}

#     # Food
#     r_food = fitbit_get(f"/1/user/-/foods/log/date/{today}.json", access_token)
#     food_data = r_food.json() if r_food.status_code == 200 else {}

#     # Heart
#     r_hr = fitbit_get(f"/1/user/-/activities/heart/date/{today}/1d.json", access_token)
#     hr_data = r_hr.json() if r_hr.status_code == 200 else {}

#     # Extract useful fields
#     steps = activities.get("summary", {}).get("steps", 0)
#     calories_burned = activities.get("summary", {}).get("caloriesOut", 0)
#     resting_hr = hr_data.get("activities-heart", [{}])[0].get("value", {}).get("restingHeartRate", 0)
#     sleep_hours = sum([s.get("minutesAsleep", 0) for s in sleep_data.get("sleep", [])]) / 60
#     calories_consumed = food_data.get("summary", {}).get("calories", 0)
#     carbs_g = food_data.get("summary", {}).get("carbs", 0)
#     protein_g = food_data.get("summary", {}).get("protein", 0)
#     fat_g = food_data.get("summary", {}).get("fat", 0)

#     daily_data = {
#         "steps": steps,
#         "calories_burned": calories_burned,
#         "resting_hr": resting_hr,
#         "sleep_hours": sleep_hours,
#         "calories_consumed": calories_consumed,
#         "carbs_g": carbs_g,
#         "protein_g": protein_g,
#         "fat_g": fat_g,
#     }
#     return daily_data


# # ---------------- Recommendation endpoint ----------------
# class RowIn(BaseModel):
#     row: dict

# @app.post("/recommend")
# def recommend_endpoint(payload: RowIn):
#     row = payload.row
#     try:
#         rec = model_recommend(row)
#         return rec
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # Health check
# @app.get("/")
# def root():
#     return {"ok": True}
# @app.get("/recommend/daily/{user_id}")
# def recommend_daily(user_id: str):
#     daily_data = fetch_daily(user_id)  # reuse the function above
#     rec = model_recommend(daily_data)
#     return {"features": daily_data, "recommendation": rec}

# api.py
import os
import json
from pathlib import Path
from datetime import date
from typing import Dict
import requests
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from urllib.parse import quote

from dotenv import load_dotenv
from recommend import recommend as model_recommend
load_dotenv()

CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8000/callback")
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"

TOKENS_FILE = Path("tokens.json")

app = FastAPI(title="Fitbit OAuth Demo")
TOKENS = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
FITBIT_BASE = "https://api.fitbit.com/1/user"
SCOPES = ["activity", "heartrate", "sleep", "nutrition"]

# ---------------- Token storage helpers ----------------
def load_tokens() -> Dict:
    if TOKENS_FILE.exists():
        return json.loads(TOKENS_FILE.read_text())
    return {}

def save_tokens(all_tokens: Dict):
    TOKENS_FILE.write_text(json.dumps(all_tokens, indent=2))

def store_user_tokens(fitbit_user_id: str, access_token: str, refresh_token: str):
    tokens = load_tokens()
    tokens[fitbit_user_id] = {
        "access_token": access_token,
        "refresh_token": refresh_token
    }
    save_tokens(tokens)

def get_user_tokens(fitbit_user_id: str):
    return load_tokens().get(fitbit_user_id)

def refresh_access_token_for_user(fitbit_user_id: str):
    tokens = get_user_tokens(fitbit_user_id)
    if not tokens:
        raise HTTPException(status_code=404, detail="User not connected")

    refresh_token = tokens["refresh_token"]
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLIENT_ID,
        },
        auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Refresh failed: {resp.text}")

    data = resp.json()
    store_user_tokens(data["user_id"], data["access_token"], data["refresh_token"])
    return data
def fitbit_get(access_token: str, endpoint: str):
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(f"{FITBIT_BASE}/-/"+endpoint, headers=headers)
    if not r.ok:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()
# ---------------- OAuth routes ----------------
@app.get("/login")
def login():
    scope = "%20".join(SCOPES)
    encoded_redirect = quote(REDIRECT_URI, safe="")
    url = (
        f"{AUTHORIZE_URL}?response_type=code&client_id={CLIENT_ID}"
        f"&redirect_uri={encoded_redirect}&scope={scope}"
    )
    return RedirectResponse(url)

@app.get("/callback")
def callback(request: Request, code: str = None, error: str = None):
    if error:
        return HTMLResponse(f"<h3>Auth failed: {error}</h3>")

    if code is None:
        return HTMLResponse("<h3>No code provided by Fitbit</h3>")

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_id": CLIENT_ID,
            "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
            "code": code
        },
        auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    if resp.status_code != 200:
        return HTMLResponse(f"<h3>Token exchange failed</h3><pre>{resp.text}</pre>", status_code=400)

    data = resp.json()
    user_id = data.get("user_id") or data.get("user", {}).get("encodedId") or "unknown"
    store_user_tokens(user_id, data["access_token"], data["refresh_token"])

    html = f"""
    <h3>Fitbit connected ✅</h3>
    <p>User id: <b>{user_id}</b></p>
    <p>Close this tab and return to the Streamlit app. Use this user id to fetch your data.</p>
    """
    return HTMLResponse(html)

@app.get("/connected")
def connected_users():
    return JSONResponse(list(load_tokens().keys()))

# ---------------- Fitbit API helpers ----------------
def fitbit_get(path: str, access_token: str):
    url = f"https://api.fitbit.com{path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    return requests.get(url, headers=headers)

# ---------------- Fetch daily data ----------------
@app.get("/fetch/daily/{user_id}")
def fetch_daily(user_id: str):
    tokens = get_user_tokens(user_id)
    if not tokens:
        raise HTTPException(status_code=404, detail="User not connected")

    access_token = tokens["access_token"]
    today = date.today().strftime("%Y-%m-%d")

    # Activities
    r = fitbit_get(f"/1/user/-/activities/date/{today}.json", access_token)
    if r.status_code == 401:
        refresh_access_token_for_user(user_id)
        tokens = get_user_tokens(user_id)
        access_token = tokens["access_token"]
        r = fitbit_get(f"/1/user/-/activities/date/{today}.json", access_token)

    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    activities = r.json()

    # Sleep
    r_sleep = fitbit_get(f"/1.2/user/-/sleep/date/{today}.json", access_token)
    sleep_data = r_sleep.json() if r_sleep.status_code == 200 else {}

    # Food
    r_food = fitbit_get(f"/1/user/-/foods/log/date/{today}.json", access_token)
    food_data = r_food.json() if r_food.status_code == 200 else {}

    # Heart
    r_hr = fitbit_get(f"/1/user/-/activities/heart/date/{today}/1d.json", access_token)
    hr_data = r_hr.json() if r_hr.status_code == 200 else {}

    steps = activities.get("summary", {}).get("steps", 0)
    calories_burned = activities.get("summary", {}).get("caloriesOut", 0)
    resting_hr = hr_data.get("activities-heart", [{}])[0].get("value", {}).get("restingHeartRate", 0)
    sleep_hours = sum([s.get("minutesAsleep", 0) for s in sleep_data.get("sleep", [])]) / 60
    calories_consumed = food_data.get("summary", {}).get("calories", 0)
    carbs_g = food_data.get("summary", {}).get("carbs", 0)
    protein_g = food_data.get("summary", {}).get("protein", 0)
    fat_g = food_data.get("summary", {}).get("fat", 0)

    return {
        "steps": steps,
        "calories_burned": calories_burned,
        "resting_hr": resting_hr,
        "sleep_hours": sleep_hours,
        "calories_consumed": calories_consumed,
        "carbs_g": carbs_g,
        "protein_g": protein_g,
        "fat_g": fat_g,
    }
# @app.get("/fetch/daily/{user_id}")
# def fetch_daily(user_id: str):
#     if user_id not in TOKENS:
#         raise HTTPException(status_code=404, detail="User not connected")

#     access_token = TOKENS[user_id]["access_token"]

#     try:
#         # 1. Activities (steps, calories)
#         activities = fitbit_get(access_token, "activities/date/today.json")

#         # 2. Sleep
#         sleep = fitbit_get(access_token, "sleep/date/today.json")

#         # 3. Heart
#         heart = fitbit_get(access_token, "activities/heart/date/today/1d.json")

#         # 4. Food
#         food = fitbit_get(access_token, "foods/log/date/today.json")

#         # Merge into one payload
#         merged = {
#             "summary": activities.get("summary", {}),
#             "sleep": sleep,
#             "hr": heart,
#             "foods": food
#         }
#         return merged

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
# ---------------- Recommendation endpoints ----------------
class RowIn(BaseModel):
    row: dict

@app.post("/recommend")
def recommend_endpoint(payload: RowIn):
    try:
        return model_recommend(payload.row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/daily/{user_id}")
def recommend_daily(user_id: str):
    try:
        daily_data = fetch_daily(user_id)
        rec = model_recommend(daily_data)
        return {"features": daily_data, "recommendation": rec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check
@app.get("/")
def root():
    return {"ok": True}
