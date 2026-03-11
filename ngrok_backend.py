"""Launch the QueryClip FastAPI backend with a persistent ngrok tunnel.

Usage:
    python ngrok_backend.py

The static ngrok domain is always https://great-repeatedly-alien.ngrok-free.app.
Set NGROK_AUTH_TOKEN in your .env file to authenticate with ngrok.
"""

import os
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

load_dotenv()

_PORT = 8000
_NGROK_DOMAIN = "great-repeatedly-alien.ngrok-free.app"
_NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")


def _start_ngrok() -> None:
    """Establish a persistent ngrok HTTP tunnel using the static domain."""
    try:
        from pyngrok import ngrok

        if _NGROK_AUTH_TOKEN:
            ngrok.set_auth_token(_NGROK_AUTH_TOKEN)
        else:
            print(
                "Warning: NGROK_AUTH_TOKEN is not set. "
                "Static domains require authentication. Add it to your .env file."
            )

        # pyngrok passes extra kwargs directly to ngrok's tunnel config
        listener = ngrok.connect(_PORT, hostname=_NGROK_DOMAIN)
        public_url = f"https://{_NGROK_DOMAIN}"

        print(f"ngrok tunnel : {public_url}")
        print(f"Local backend: http://localhost:{_PORT}")

        Path("ngrok_url.txt").write_text(public_url)

    except ImportError:
        print("pyngrok is not installed. Run: pip install pyngrok")
        sys.exit(1)
    except Exception as exc:
        print(f"Warning: Could not start ngrok tunnel: {exc}")
        print(f"The API will only be available at http://localhost:{_PORT}")


if __name__ == "__main__":
    print("Starting QueryClip backend...")
    _start_ngrok()

    uvicorn.run(
        "backend.fastapi_app:app",
        host="0.0.0.0",
        port=_PORT,
        reload=False,
        log_level="info",
    )

