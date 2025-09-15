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

# Load model + vectorizer + intents
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("chatbot_intents.json", "r") as f:
    intents = json.load(f)

API_KEY = "8a611495b6b56f082f045d2ffed3389c"


#Weather
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


#World Time
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


#Universal Time
def get_universal_time():
    now = datetime.utcnow()
    return f"The current Universal (UTC) time is {now.strftime('%Y-%m-%d %H:%M:%S')} 🌍"


#Chit-chat Logic
chit_chat_responses = {
    "how are you": "I’m fine, and you? 🙂",
    "good, you": "I’m good too! What can I help you with? 😎",
    "what can you do": f"I can greet you, tell you my name, give you the time, check the weather in different cities, and chat a little 😊",
    "let me ask you a question": "Sure, go ahead! 👂",
    "who are you": f"I'm {BOT_NAME}, your friendly assistant 🤖",
    "what is your name": f"My name is {BOT_NAME}!",
}


#Response Generator
def generate_response(user_input):
    global last_intent

    # Normalize input for chit-chat
    normalized = user_input.lower().strip()

    for key, reply in chit_chat_responses.items():
        if key in normalized:
            return reply

    # Split into possible parts (for multiple questions in one sentence)
    parts = re.split(r"\s*(?:\?|\.|!|,|\band\b)\s*", user_input, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]

    responses = []
    answered_intent = set()

    for part in parts:
        try:
            X_test = vectorizer.transform([part])
            proba = model.predict_proba(X_test)[0]
            max_idx = proba.argmax()
            confidence = proba[max_idx]
            tag = model.classes_[max_idx]

            # Apply confidence threshold
            if confidence < 0.75:
                responses.append("I didn’t quite get that, could you say it another way? 🤔")
                continue
        except Exception:
            responses.append("I’m not sure I understand 🤔")
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
            for intent in intents["intents"]:
                if intent["tag"] == tag:
                    responses.append(random.choice(intent["responses"]))
                    break

        answered_intent.add(tag)

    return " ".join(responses) if responses else "I didn’t catch that. Could you rephrase? 🤔"


#Chat Route
last_intent = None

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    reply = generate_response(user_input)
    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(debug=True)
