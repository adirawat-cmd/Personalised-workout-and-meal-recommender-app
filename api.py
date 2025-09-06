# api.py
import os
import json
from typing import Dict
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
# at top of api.py (after other imports)
from recommend import recommend as model_recommend

# add to api.py
from pydantic import BaseModel
from urllib.parse import quote

load_dotenv()

CLIENT_ID = os.getenv("FITBIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("FITBIT_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://127.0.0.1:8000/callback")
TOKEN_URL = "https://api.fitbit.com/oauth2/token"
AUTHORIZE_URL = "https://www.fitbit.com/oauth2/authorize"

# Where we persist tokens locally for demo. Use a DB in production.
TOKENS_FILE = Path("tokens.json")

app = FastAPI(title="Fitbit OAuth Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCOPES = ["activity", "heartrate", "sleep", "nutrition"]  # adjust as needed

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
    # store new tokens
    store_user_tokens(data["user_id"], data["access_token"], data["refresh_token"])
    return data

# @app.get("/login")
# def login():
#     """Redirect user to Fitbit auth page."""
#     scope = "%20".join(SCOPES)
#     url = (
#         f"{AUTHORIZE_URL}?response_type=code&client_id={CLIENT_ID}"
#         f"&redirect_uri={REDIRECT_URI}&scope={scope}"
#     )
#     return RedirectResponse(url)
@app.get("/login")
def login():
    """Redirect user to Fitbit auth page."""
    scope = "%20".join(SCOPES)
    encoded_redirect = quote(REDIRECT_URI, safe="")
    url = (
        f"{AUTHORIZE_URL}?response_type=code&client_id={CLIENT_ID}"
        f"&redirect_uri={encoded_redirect}&scope={scope}"
    )
    return RedirectResponse(url)

@app.get("/callback")
def callback(request: Request, code: str = None, error: str = None):
    """Fitbit will redirect here with ?code=..."""
    if error:
        return HTMLResponse(f"<h3>Auth failed: {error}</h3>")

    if code is None:
        return HTMLResponse("<h3>No code provided by Fitbit</h3>")

    # Exchange authorization code for tokens
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
    # data includes access_token, refresh_token, user_id
    user_id = data.get("user_id") or data.get("user", {}).get("encodedId") or "unknown"
    store_user_tokens(user_id, data["access_token"], data["refresh_token"])

    # Return a simple page with the user's id and instructions
    html = f"""
    <h3>Fitbit connected ✅</h3>
    <p>User id: <b>{user_id}</b></p>
    <p>Close this tab and go back to the Streamlit app. Use this user id to fetch your data.</p>
    """
    return HTMLResponse(html)

@app.get("/connected")
def connected_users():
    """Return list of connected Fitbit user_ids."""
    tokens = load_tokens()
    return JSONResponse(list(tokens.keys()))

def fitbit_get(path: str, access_token: str, params: dict = None):
    url = f"https://api.fitbit.com{path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.get(url, headers=headers, params=params or {})
    return r

# @app.get("/fetch/daily/{user_id}")
# def fetch_daily(user_id: str):
#     """Fetch aggregated daily summary (steps, calories, sleep) for today for given user_id."""
#     tokens = get_user_tokens(user_id)
#     if not tokens:
#         raise HTTPException(status_code=404, detail="User not connected")

#     access_token = tokens["access_token"]
#     print(f"Fetching daily summary for user_id={user_id}, token={access_token[:10]}...")
#     # try request; if 401, refresh and retry
#     r = fitbit_get("/1/user/-/activities/date/today.json", access_token)
#     print("Activities response:", r.status_code, r.text[:300])
#     if r.status_code == 401:
#         refresh_access_token_for_user(user_id)
#         tokens = get_user_tokens(user_id)
#         access_token = tokens["access_token"]
#         r = fitbit_get("/1/user/-/activities/date/today.json", access_token)

#     if r.status_code != 200:
#         raise HTTPException(status_code=r.status_code, detail=r.text)

#     activities = r.json()
#     # Another endpoints: sleep summary, foods, heart
#     r_sleep = fitbit_get("/1.2/user/-/sleep/date/today.json", access_token)
#     r_food = fitbit_get("/1/user/-/foods/log/date/today.json", access_token)
#     r_hr = fitbit_get("/1/user/-/activities/heart/date/today/1d.json", access_token)

#     # combine safely
#     out = {"activities": activities}
#     out["sleep"] = r_sleep.json() if r_sleep.status_code == 200 else {}
#     out["foods"] = r_food.json() if r_food.status_code == 200 else {}
#     out["hr"] = r_hr.json() if r_hr.status_code == 200 else {}


#     return out
from datetime import datetime

@app.get("/fetch/daily/{user_id}")
def fetch_daily(user_id: str):
    tokens = get_user_tokens(user_id)
    if not tokens:
        raise HTTPException(status_code=404, detail="User not connected")

    access_token = tokens["access_token"]

    # Fitbit requires explicit YYYY-MM-DD, not "today"
    today = datetime.today().strftime("%Y-%m-%d")

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
    # Food
    r_food = fitbit_get(f"/1/user/-/foods/log/date/{today}.json", access_token)
    # Heart
    r_hr = fitbit_get(f"/1/user/-/activities/heart/date/{today}/1d.json", access_token)

    out = {"activities": activities}
    out["sleep"] = r_sleep.json() if r_sleep.status_code == 200 else {}
    out["foods"] = r_food.json() if r_food.status_code == 200 else {}
    out["hr"] = r_hr.json() if r_hr.status_code == 200 else {}

    return out


# lightweight health-check
@app.get("/")
def root():
    return {"ok": True}


class RowIn(BaseModel):
    row: dict

@app.post("/recommend")
def recommend_endpoint(payload: RowIn):
    row = payload.row
    try:
        rec = model_recommend(row)
        return rec
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
