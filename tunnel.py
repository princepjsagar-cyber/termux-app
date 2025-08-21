import time
import sys
import os
from pyngrok import ngrok, conf


def main() -> None:
    conf.get_default().monitor_thread = False

    port_str = os.environ.get("TUNNEL_PORT", "8000")
    try:
        port = int(port_str)
    except ValueError:
        port = 8000

    # Start HTTP tunnel to local dashboard on the given port
    tunnel = ngrok.connect(port, "http")

    public_url = tunnel.public_url
    # Write the public URL to a well-known file for other processes to read
    out_path = "/workspace/.public_dashboard_url"
    try:
        with open(out_path, "w") as f:
            f.write(public_url)
    except Exception:
        pass

    print(public_url, flush=True)

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        try:
            ngrok.disconnect(public_url)
        finally:
            ngrok.kill()


if __name__ == "__main__":
    main()

