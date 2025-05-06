import streamlit as st
import requests
import os
import json
from typing import Iterator

# This is the main Streamlit app for cloud deployment
# It will connect to the Ngrok URL for the backend

# Read the Ngrok URL from secrets or environment
NGROK_URL = st.secrets.get("NGROK_URL", os.getenv("NGROK_URL", ""))

# If running locally and we have a saved URL, use that
if not NGROK_URL and os.path.exists("ngrok_url.txt"):
    with open("ngrok_url.txt", "r") as f:
        NGROK_URL = f.read().strip()

if not NGROK_URL:
    st.error("No Ngrok URL found! The backend server is not accessible.")
    st.info("You need to run ngrok_backend.py first and set the URL in Streamlit secrets.")
    st.stop()

# API endpoints
API_URL = f"{NGROK_URL}/query"
HEALTH_CHECK_URL = f"{NGROK_URL}/"

# Keep most of your existing streamlit_with_error_handling.py code
# But replace the localhost URLs with the Ngrok URL

def is_api_available():
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        return response.status_code == 200
    except:
        return False

# Rest of your Streamlit app code...
# (Include all your existing UI and functionality from streamlit_with_error_handling.py)