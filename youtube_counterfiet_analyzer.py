# ==================================================
# PRAMAAN – Counterfeit Risk Analyzer (Single File)
# ==================================================

import os
import json
import requests
import pandas as pd
import streamlit as st
#from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from openai import OpenAI

# ==================================================
# 1. ENV SETUP
# ==================================================
#load_dotenv()
#Load secrets
def load_secrets(file_path="secrets.txt"):
    secrets = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.split("=", 1)
            secrets[key.strip()] = value.strip()
    return secrets

#secrets = load_secrets()
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
YOUTUBE_TRANSCRIPT_API_KEY = st.secrets["YOUTUBE_TRANSCRIPT_API_KEY"]

#YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#YOUTUBE_TRANSCRIPT_API_KEY = os.getenv("YOUTUBE_TRANSCRIPT_API_KEY")

if not YOUTUBE_API_KEY:
    raise ValueError("Missing YOUTUBE_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")
if not YOUTUBE_TRANSCRIPT_API_KEY:
    raise ValueError("Missing YOUTUBE_TRANSCRIPT_API_KEY")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = "gpt-4.1-mini"
YOUTUBE_TRANSCRIPT_API_URL = "https://www.youtube-transcript.io/api/transcript"

# ==================================================
# 2. HELPERS
# ==================================================
def extract_video_id(value: str):
    if "youtube.com" in value or "youtu.be" in value:
        parsed = urlparse(value)
        if parsed.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.hostname == "youtu.be":
            return parsed.path.lstrip("/")
        return None
    return value.strip()


def fetch_transcript(video_id: str) -> str:
    try:
        response = requests.get(
            YOUTUBE_TRANSCRIPT_API_URL,
            headers={
                "Authorization": f"Basic {YOUTUBE_TRANSCRIPT_API_KEY}"
            },
            params={"video_id": video_id},
            timeout=15
        )

        if response.status_code == 404:
            return ""

        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "text" in data:
            return data["text"].strip()

        if isinstance(data, list) and data and "text" in data[0]:
            return data[0]["text"].strip()

        transcript_chunks = data[0]["tracks"][0]["transcript"]
        return " ".join(
            c.get("text", "") for c in transcript_chunks if c.get("text")
        ).strip()

    except Exception:
        return ""


def get_video_metadata(video_id: str):
    response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    ).execute()

    if not response.get("items"):
        return None

    item = response["items"][0]
    s = item.get("snippet", {})
    stt = item.get("statistics", {})
    c = item.get("contentDetails", {})

    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": s.get("title", ""),
        "description": s.get("description", ""),
        "tags": "|".join(s.get("tags", [])),
        "channel_title": s.get("channelTitle", ""),
        "published_at": s.get("publishedAt", ""),
        "view_count": int(stt.get("viewCount", 0)),
        "like_count": int(stt.get("likeCount", 0)) if "likeCount" in stt else None,
        "comment_count": int(stt.get("commentCount", 0)) if "commentCount" in stt else None,
        "duration": c.get("duration", "")
    }


def analyze_video_with_gpt(title, keywords, transcript, description):
    has_transcript = len(transcript.strip()) > 40

    system_prompt = """
You are a precise product authenticity and counterfeit risk analyst.
Analyze YouTube video metadata to estimate counterfeit risk.

Respond ONLY in strict JSON:
{
  "counterfeit_risk_percent": number,
  "suspected_is_counterfeit": boolean,
  "reasoning": string,
  "brand_models_mentioned": string[],
  "price_points_mentioned_in_inr": number[],
  "red_flags": string[],
  "green_flags": string[]
}
"""

    user_prompt = f"""
Title:
{title}

Tags:
{keywords}

Description:
{description}

Transcript available: {"yes" if has_transcript else "no"}

Transcript:
\"\"\"{transcript if has_transcript else ""}\"\"\"
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return {
            "counterfeit_risk_percent": None,
            "suspected_is_counterfeit": None,
            "reasoning": "JSON parse error",
            "brand_models_mentioned": [],
            "price_points_mentioned_in_inr": [],
            "red_flags": [],
            "green_flags": []
        }

# ==================================================
# 3. STREAMLIT UI
# ==================================================

st.set_page_config(
    page_title="PRAMAAN – AI-Based Counterfeit Risk Analysis for YouTube Videos",
    layout="centered"
)


st.markdown("""
## **PRAMAAN – AI-Based Counterfeit Risk Analysis for YouTube Videos**
*Platform for Reliability Assessment & Counterfeit Monitoring*

Research-driven analysis of counterfeit risk in YouTube videos using AI.
""")




youtube_url = st.text_input(
    "Paste YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=XXXXXXXX"
)

if st.button("Analyze Video"):
    if not youtube_url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(youtube_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
        st.stop()

    with st.spinner("Fetching video metadata..."):
        video = get_video_metadata(video_id)

    if not video:
        st.error("Video not found.")
        st.stop()

    with st.spinner("Fetching transcript (if available)..."):
        transcript = fetch_transcript(video_id)

    with st.spinner("Analyzing counterfeit risk..."):
        analysis = analyze_video_with_gpt(
            title=video["title"],
            keywords=video["tags"],
            transcript=transcript,
            description=video["description"]
        )

    st.subheader("🔍 Counterfeit Risk Assessment")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Counterfeit Risk %", f"{analysis['counterfeit_risk_percent']}%")
    with col2:
        st.metric(
            "Suspected Counterfeit",
            "Yes" if analysis["suspected_is_counterfeit"] else "No"
        )

    st.subheader("🧠 Analysis Details")

    st.markdown("**Reasoning**")
    st.write(analysis["reasoning"])

    st.markdown("**Brands / Models Mentioned**")
    st.write(", ".join(analysis["brand_models_mentioned"]) or "None detected")

    st.markdown("**Price Points Mentioned (INR)**")
    st.write(
        ", ".join(str(x) for x in analysis["price_points_mentioned_in_inr"])
        or "None detected"
    )

    st.markdown("**🚩 Red Flags**")
    if analysis["red_flags"]:
        for rf in analysis["red_flags"]:
            st.write(f"- {rf}")
    else:
        st.write("None detected")

    st.markdown("**✅ Green Flags**")
    if analysis["green_flags"]:
        for gf in analysis["green_flags"]:
            st.write(f"- {gf}")
    else:
        st.write("None detected")


    with st.expander("📄 View Transcript"):
        if transcript:
            st.write(transcript)
        else:
            st.info("Transcript not available.")

    with st.expander("🧠 Raw AI Output (JSON)"):
        st.json(analysis)
