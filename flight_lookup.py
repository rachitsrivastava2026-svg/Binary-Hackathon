"""
flight_lookup.py — real flight offers via Duffel's live flight search API.

Duffel (https://duffel.com) is a genuine, free, self-serve REST API for
flight search (and booking, though this module only searches):
free signup, a real test mode (test tokens start with "duffel_test_",
no money ever moves), covers a huge network of airlines and routes.
It returns real airline names, departure/arrival times, and prices — not
mock data.

Setup:
    1. Sign up free (no card) at https://duffel.com
    2. Grab your test API key from the dashboard (starts with duffel_test_)
    3. export DUFFEL_API_KEY=duffel_test_...   (or put it in your .env file)

Usage:
    python flight_lookup.py BLR DEL
    python flight_lookup.py BLR DEL --date 2026-10-05
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

import requests

DUFFEL_API_KEY = os.environ.get("DUFFEL_API_KEY")
BASE_URL = "https://api.duffel.com"

# Rough, FIXED exchange rates, used only to show an approximate
# INR-equivalent price when Duffel returns an offer priced in another
# currency. These are not live rates — swap in a real FX API if you need
# accuracy for anything beyond a demo.
FX_TO_INR = {"INR": 1.0, "USD": 83.0, "GBP": 105.0, "EUR": 90.0}


def _headers(version="v2"):
    return {
        "Authorization": f"Bearer {DUFFEL_API_KEY}",
        "Duffel-Version": version,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def search_real_flights(origin: str, destination: str, date: str = None, limit: int = 10):
    """Search real flight offers between two airports via Duffel.

    Args:
        origin: 3-letter IATA airport code, e.g. "BLR"
        destination: 3-letter IATA airport code, e.g. "DEL"
        date: optional "YYYY-MM-DD" departure date. Duffel requires a
            date per search — if omitted, defaults to ~2 weeks out.
        limit: max number of offers to return

    Returns:
        {"status": "ok", "count": N, "flights": [{"flight_id", "airline",
         "origin", "destination", "date", "departure_time", "arrival_time",
         "price_inr", "price_original", "seats_available"}, ...]}
        or {"status": "error", "message": "..."} if the key is missing / call fails
    """
    if not DUFFEL_API_KEY:
        return {
            "status": "error",
            "message": (
                "DUFFEL_API_KEY is not set. Get a free test key (no card) at "
                "https://duffel.com, then export DUFFEL_API_KEY=duffel_test_..."
            ),
        }

    if not date:
        # Duffel requires a departure_date per slice — default to a
        # couple weeks out if the caller didn't give one.
        date = (datetime.now().date() + timedelta(days=14)).isoformat()

    body = {
        "data": {
            "slices": [{
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": date,
            }],
            "passengers": [{"type": "adult"}],
            "cabin_class": "economy",
        }
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/air/offer_requests?return_offers=true",
            headers=_headers(),
            json=body,
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {e}"}

    offers = payload.get("data", {}).get("offers", [])[:limit]

    flights = []
    for o in offers:
        slice0 = o["slices"][0]
        first_seg = slice0["segments"][0]
        last_seg = slice0["segments"][-1]
        currency = o["total_currency"]
        amount = float(o["total_amount"])
        flights.append({
            "flight_id": o["id"],
            "airline": o["owner"]["name"],
            "origin": first_seg["origin"]["iata_code"],
            "destination": last_seg["destination"]["iata_code"],
            "date": first_seg["departing_at"][:10],
            "departure_time": first_seg["departing_at"][11:16],
            "arrival_time": last_seg["arriving_at"][11:16],
            "price_inr": round(amount * FX_TO_INR.get(currency, 1.0)),
            "price_original": f"{amount} {currency}",  # transparency re: the FX estimate above
            # Duffel offers don't expose a literal remaining-seat count —
            # if it came back as an offer, treat it as bookable.
            "seats_available": 9,
        })

    return {"status": "ok", "count": len(flights), "flights": flights}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python flight_lookup.py ORIGIN DEST [--date YYYY-MM-DD]")
        sys.exit(1)

    origin_arg = sys.argv[1]
    dest_arg = sys.argv[2]
    date_arg = None
    if "--date" in sys.argv:
        idx = sys.argv.index("--date")
        date_arg = sys.argv[idx + 1]

    result = search_real_flights(origin_arg, dest_arg, date=date_arg)

    if result["status"] == "error":
        print("Error:", result["message"])
    else:
        print(f"Found {result['count']} flights {origin_arg.upper()} -> {dest_arg.upper()}:\n")
        for f in result["flights"]:
            print(f"  {f['airline']} — {f['departure_time']} to {f['arrival_time']} "
                  f"on {f['date']} | ₹{f['price_inr']} ({f['price_original']})")
