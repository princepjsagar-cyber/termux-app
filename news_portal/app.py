import os
import requests
from flask import Flask, render_template, jsonify


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    api_key = os.environ.get("NEWS_API_KEY", "").strip()
    news_url = os.environ.get("NEWS_API_ENDPOINT", "https://newsapi.org/v2/top-headlines").strip()

    @app.route("/")
    def home():
        return render_template("index.html")

    @app.route("/get_news")
    def get_news():
        params = {
            "country": os.environ.get("NEWS_COUNTRY", "in"),
            "apiKey": api_key,
            "pageSize": int(os.environ.get("NEWS_MAX_RESULTS", "10") or 10),
        }
        try:
            resp = requests.get(news_url, params=params, timeout=15)
            resp.raise_for_status()
            return jsonify(resp.json())
        except requests.RequestException as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    # Bind to all interfaces; production can put behind a reverse proxy
    create_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False, threaded=True)

