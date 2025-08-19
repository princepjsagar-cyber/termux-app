import os
import sqlite3
import uuid
from flask import Flask, request, jsonify


DB_PATH = os.environ.get("REFERRAL_DB_PATH", "/workspace/referral/users.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    try:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                referral_link TEXT NOT NULL UNIQUE,
                rewards INTEGER DEFAULT 0
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/register")
    def register_user():
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        if not username:
            return jsonify({"error": "Username is required"}), 400
        referral_link = str(uuid.uuid4())
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (username, referral_link) VALUES (?, ?)",
                (username, referral_link),
            )
            conn.commit()
        finally:
            conn.close()
        return jsonify({"username": username, "referral_link": referral_link}), 201

    @app.post("/reward")
    def reward_user():
        data = request.get_json(silent=True) or {}
        referral_link = data.get("referral_link")
        if not referral_link:
            return jsonify({"error": "Referral link is required"}), 400
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute(
                "UPDATE users SET rewards = rewards + 1 WHERE referral_link = ?",
                (referral_link,),
            )
            conn.commit()
        finally:
            conn.close()
        return jsonify({"message": "Reward added successfully"}), 200

    @app.get("/user")
    def get_user_by_ref():
        ref = request.args.get("ref")
        if not ref:
            return jsonify({"error": "ref is required"}), 400
        conn = get_conn()
        try:
            c = conn.cursor()
            c.execute(
                "SELECT id, username, referral_link, rewards FROM users WHERE referral_link = ?",
                (ref,),
            )
            row = c.fetchone()
        finally:
            conn.close()
        if not row:
            return jsonify({"error": "not found"}), 404
        return jsonify({"id": row[0], "username": row[1], "referral_link": row[2], "rewards": row[3]})

    return app


if __name__ == "__main__":
    init_db()
    create_app().run(host="0.0.0.0", port=int(os.environ.get("REFERRAL_PORT", "8090")), debug=False)

