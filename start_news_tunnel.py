import os
import time
from pyngrok import ngrok, conf


def main():
    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        print("NGROK_AUTHTOKEN not set; skipping tunnel")
        return

    # Configure and connect
    conf.get_default().auth_token = token
    # Bind to local Flask app port
    public_url = ngrok.connect(addr=8080, proto="http").public_url
    print("news public:", public_url)
    # Write to file for bot command to read
    path = "/workspace/news_portal_url.txt"
    with open(path, "w") as f:
        f.write(public_url)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

