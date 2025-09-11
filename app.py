from flask import Flask, request, jsonify
import json
import pickle
import random
import pytz
from datetime import datetime
import re
import requests
from flask_cors import CORS
import difflib

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins":["https://adebola-stephen.github.io"]}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://adebola-stephen.github.io"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response



# ---------------- LOAD DATA ----------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_intents.json", "r") as f:
    intents = json.load(f)

API_KEY = "8a611495b6b56f082f045d2ffed3389c"


# ---------------- WEATHER ----------------
def get_weather(user_input, default_city="Lagos"):
    match = re.search(r'weather in ([a-zA-Z\s]+)', user_input.lower())
    city = match.group(1).strip().title() if match else default_city

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"The weather in {city} is {desc} with {temp}°C 🌤️"
    else:
        return f"Sorry, I couldn’t fetch the weather for {city} 🌧️"


# ---------------- WORLD TIME ----------------
def get_world_time(location_name):
    try:
        url = "http://worldtimeapi.org/api/timezone"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        timezones = response.json()

        candidates = [tz.split("/")[-1] for tz in timezones]
        match = difflib.get_close_matches(location_name.title(), candidates, n=1, cutoff=0.6)

        if not match:
            return f"Sorry, I couldn't determine the timezone for {location_name} ⏰"

        full_tz = next(tz for tz in timezones if tz.endswith(match[0]))
        tz_url = f"http://worldtimeapi.org/api/timezone/{full_tz}"
        tz_response = requests.get(tz_url, timeout=5)
        tz_response.raise_for_status()
        tz_data = tz_response.json()

        datetime_str = tz_data.get("datetime")
        if not datetime_str:
            return f"Time data not available for {location_name} ⏰"

        date, time_part = datetime_str.split("T")
        time_clean = time_part.split(".")[0]

        return f"The current time in {location_name.title()} is {date} {time_clean} ⏰"

    except requests.exceptions.Timeout:
        return f"Request timed out while fetching time for {location_name} ❌"
    except requests.exceptions.RequestException as e:
        return f"Error fetching time for {location_name}: {e}"
    except Exception:
        return f"Sorry, I couldn't fetch the time for {location_name} ⏰"


# ---------------- UNIVERSAL TIME ----------------
def get_universal_time():
    now = datetime.utcnow()
    return f"The current Universal (UTC) time is {now.strftime('%Y-%m-%d %H:%M:%S')} 🌍"


# ---------------- CHAT ROUTE ----------------
last_intent = None
conversation_state = {"greeting_step": 0}


@app.route("/chat", methods=["POST"])
def chat():
    global last_intent, conversation_state
    user_input = request.json.get("message", "")

    if not user_input:
        return jsonify({"response": "Please say something!"})

    parts = re.split(r"\s*(?:\?|\.|!|,|\band\b)\s*", user_input, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    responses = []
    answered_intent = set()
    context_triggered = False

    for part in parts:
        if context_triggered:
            continue

        try:
            X_test = vectorizer.transform([part])
            tag = model.predict(X_test)[0]
        except Exception:
            responses.append("I’m not sure I understand 🤔")
            continue

        if last_intent == "greeting" and ("fine" in part.lower() or "good" in part.lower()):
            responses.append("That's good to hear! What can I do for you today? 😊")
            last_intent = "smalltalk"
            context_triggered = True
            continue

        if tag in answered_intent:
            continue

        if tag == "time":
            if "universal" in part.lower() or "utc" in part.lower():
                responses.append(get_universal_time())
            elif "in" in part.lower():
                city = part.split("in")[-1].strip()
                responses.append(get_world_time(city))
            else:
                now = datetime.now()
                responses.append(f"The current local time is {now.strftime('%H:%M:%S')} ⏰")

        elif tag == "weather":
            responses.append(get_weather(part))

        else:
            matched = False
            for intent in intents["intents"]:
                if intent["tag"] == tag:
                    if intent["responses"]:
                        responses.append("Hello! How are you today? 😊" if tag == "greeting" else random.choice(intent["responses"]))
                    else:
                        responses.append("Hmm...")
                    matched = True
                    break

            if not matched:
                responses.append("I’m not sure I understand 🤔")

        last_intent = tag
        answered_intent.add(tag)

    final_response = " ".join(responses) if responses else "I didn’t catch that. Could you rephrase? 🤔"
    return jsonify({"response": final_response})


if __name__ == "__main__":
    app.run(debug=True)
