from flask import Flask, request, jsonify
import json
import pickle
import random
from datetime import datetime
import re
import requests
from flask_cors import CORS
import difflib

app = Flask(__name__)

BOT_NAME = "Moyennn"

# Allow ONLY your frontend
CORS(app, origins=["https://adebola-stephen.github.io"])

# LOAD DATA
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_intents.json", "r") as f:
    intents = json.load(f)

API_KEY = "8a611495b6b56f082f045d2ffed3389c"


# WEATHER
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


# WORLD TIME
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

    except Exception:
        return f"Sorry, I couldn't fetch the time for {location_name} ⏰"


# UNIVERSAL TIME
def get_universal_time():
    now = datetime.utcnow()
    return f"The current Universal (UTC) time is {now.strftime('%Y-%m-%d %H:%M:%S')} 🌍"


# RULE-BASED RESPONSES (priority before ML)
def rule_based_response(user_input):
    msg = user_input.lower()

    if any(word in msg for word in ["hi", "hello", "hey"]):
        return "Hello there 👋"

    elif "how are you" in msg:
        return "I’m fine, and you? 🙂"

    elif "what can you do" in msg or "help" in msg:
        return f"I can greet you, tell you my name, give you the time, check the weather in different cities, and chat a little 😊"

    elif "your name" in msg:
        return f"My name is {BOT_NAME} 🤖"

    elif "time" in msg and ("universal" in msg or "utc" in msg):
        return get_universal_time()

    return None  # no rule matched


# RESPONSE GENERATION
def generate_response(user_input):
    global last_intent

    # First check rule-based overrides
    rule_reply = rule_based_response(user_input)
    if rule_reply:
        return rule_reply

    # Otherwise fall back to ML + intents
    parts = re.split(r"\s*(?:\?|\.|!|,|\band\b)\s*", user_input, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    responses = []
    answered_intent = set()

    for part in parts:
        try:
            X_test = vectorizer.transform([part])
            tag = model.predict(X_test)[0]
        except Exception:
            responses.append("I’m not sure I understand 🤔")
            continue

        if tag in answered_intent:
            continue

        if tag == "time":
            if "in" in part.lower():
                city = part.split("in")[-1].strip()
                responses.append(get_world_time(city))
            else:
                now = datetime.now()
                responses.append(f"The current local time is {now.strftime('%H:%M:%S')} ⏰")

        elif tag == "weather":
            responses.append(get_weather(part))

        else:
            for intent in intents["intents"]:
                if intent["tag"] == tag:
                    responses.append(random.choice(intent["responses"]))
                    break

        answered_intent.add(tag)

    return " ".join(responses) if responses else "Sorry, I didn’t quite understand that. Can you try rephrasing? 🤔"


# CHAT ROUTE
last_intent = None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    reply = generate_response(user_input)
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
