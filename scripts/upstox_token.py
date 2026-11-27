"""Generate a fresh Upstox access token with no browser login, using Upstox's TOTP token endpoint
(POST /v2/totp-login/authorization/token: API key in the x-api-key header, the account's 6-digit
PIN and a current TOTP code in the body). Upstox access tokens expire at 03:30 IST every day, so
run this before each eod/open - the GitHub workflows do.

One-time setup
  1. Upstox app/web: enable TOTP (Profile -> Security / 2FA -> TOTP) and, while setting it up, copy
     the TOTP SECRET KEY (the base32 text behind the QR code - "can't scan? enter this key").
     Add it to your authenticator app too, so you can still log in by hand.
  2. Upstox developer apps: note the app's API key.
  3. Provide three values as environment variables (GitHub: repo Settings -> Secrets and
     variables -> Actions -> New repository secret):
         UPSTOX_API_KEY      the app's API key
         UPSTOX_PIN          your 6-digit Upstox PIN
         UPSTOX_TOTP_SECRET  the base32 TOTP secret from step 1
     Together these can log in to your Upstox account: keep the repo private, never print them.

Usage
  python scripts/upstox_token.py --check             # generate a token and verify it (prints the user name only)
  python scripts/upstox_token.py --print             # print the token (local use)
  python scripts/upstox_token.py --github-env        # GitHub Actions: masked, exported to later steps
  PowerShell:  $env:UPSTOX_ACCESS_TOKEN = (python scripts\\upstox_token.py --print)

Notes
- Generating a token revokes the app's previous token (Upstox: "that token is revoked and a new
  one is issued") - a token you made by hand for the same app stops working.
- Exit codes: 0 ok, 1 missing configuration, 2 Upstox refused / network error.
- If the TOTP secrets are not configured and --optional is given, the script exits 0 without doing
  anything, so the workflow falls back to a UPSTOX_ACCESS_TOKEN secret set by hand.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import os
import struct
import sys
import time

import requests

TOKEN_URL = "https://api.upstox.com/v2/totp-login/authorization/token"
PROFILE_URL = "https://api.upstox.com/v2/user/profile"


def totp(secret_b32: str, at: float | None = None, step: int = 30, digits: int = 6) -> str:
    """RFC 6238 TOTP (HMAC-SHA1), the algorithm authenticator apps use."""
    key = secret_b32.strip().replace(" ", "").upper()
    key += "=" * (-len(key) % 8)
    raw = base64.b32decode(key)
    counter = int((time.time() if at is None else at) // step)
    mac = hmac.new(raw, struct.pack(">Q", counter), hashlib.sha1).digest()
    off = mac[-1] & 0x0F
    code = (struct.unpack(">I", mac[off:off + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


def generate(api_key: str, pin: str, secret: str, timeout: float = 20.0) -> dict:
    # a code about to roll over may be stale by the time Upstox checks it: wait for a fresh window
    if 30 - (time.time() % 30) < 5:
        time.sleep(30 - (time.time() % 30) + 0.5)
    r = requests.post(TOKEN_URL, timeout=timeout,
                      headers={"accept": "application/json", "Content-Type": "application/json",
                               "x-api-key": api_key},
                      json={"totp": totp(secret), "pin": pin})
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code >= 400:
        errs = body.get("errors") or [{}]
        msg = errs[0].get("message") or r.text[:200]
        raise RuntimeError(f"Upstox refused the TOTP login ({r.status_code}): {msg}")
    data = body.get("data", body)
    if not data.get("access_token"):
        raise RuntimeError(f"no access_token in the Upstox response (keys: {sorted(data)})")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print the access token to stdout")
    ap.add_argument("--github-env", action="store_true",
                    help="mask the token in the Actions log and export UPSTOX_ACCESS_TOKEN to later steps")
    ap.add_argument("--check", action="store_true", help="verify the new token with a profile call")
    ap.add_argument("--optional", action="store_true",
                    help="exit 0 quietly if the TOTP secrets are not configured")
    a = ap.parse_args()

    need = ("UPSTOX_API_KEY", "UPSTOX_PIN", "UPSTOX_TOTP_SECRET")
    missing = [k for k in need if not os.environ.get(k, "").strip()]
    if missing:
        if a.optional:
            print(f"upstox_token: {', '.join(missing)} not set - keeping the existing UPSTOX_ACCESS_TOKEN",
                  file=sys.stderr)
            return 0
        print(f"upstox_token: missing environment variable(s): {', '.join(missing)}", file=sys.stderr)
        return 1
    try:
        data = generate(os.environ["UPSTOX_API_KEY"].strip(), os.environ["UPSTOX_PIN"].strip(),
                        os.environ["UPSTOX_TOTP_SECRET"])
    except (RuntimeError, requests.RequestException, ValueError) as e:
        print(f"upstox_token: {e}", file=sys.stderr)
        return 2
    token = data["access_token"]

    if a.github_env:
        print(f"::add-mask::{token}")                       # hide it in every later log line
        env_file = os.environ.get("GITHUB_ENV")
        if not env_file:
            print("upstox_token: --github-env outside GitHub Actions (GITHUB_ENV not set)", file=sys.stderr)
            return 1
        with open(env_file, "a", encoding="utf-8") as f:
            f.write(f"UPSTOX_ACCESS_TOKEN={token}\n")
    if a.check:
        r = requests.get(PROFILE_URL, timeout=20,
                         headers={"Accept": "application/json", "Authorization": f"Bearer {token}"})
        if r.status_code >= 400:
            print(f"upstox_token: new token failed the profile check ({r.status_code})", file=sys.stderr)
            return 2
        prof = (r.json() or {}).get("data") or {}
        print(f"upstox_token: token OK for {prof.get('user_name') or prof.get('user_id') or 'user'}",
              file=sys.stderr)
    if a.print:
        print(token)
    if not (a.print or a.github_env or a.check):
        print("upstox_token: token generated (use --print, --check or --github-env)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
