"""
hotel_lookup.py — real hotel names + ratings for any city, via hotels-api.com

hotels-api.com is a genuine, free, self-serve REST API (no credit card):
100 requests/month on the free plan, 1M+ hotels across 200+ countries,
including full coverage of India. It returns real hotel names, star
ratings, GPS coordinates, and amenities — it does NOT return live prices,
which is fine here since that's not what was asked for.

Setup:
    1. Sign up free (no card) at https://hotels-api.com/register/
    2. Copy your API key from the console
    3. export HOTELS_API_KEY=your_key_here   (or put it in your .env file)

Usage:
    python hotel_lookup.py "Jaipur"
    python hotel_lookup.py "Mumbai" --min-rating 4
"""
import os
import sys
import requests
from dotenv import load_dotenv
load_dotenv()

HOTELS_API_KEY = os.environ.get("HOTELS_API_KEYS")
BASE_URL = "https://api.hotels-api.com/v2"


def search_real_hotels(city: str, country: str = "India", min_rating: float = None, limit: int = 10):
    """Look up real hotels in a city via hotels-api.com's free tier.

    Args:
        city: city name, e.g. "Jaipur" (full name, not an airport code)
        country: defaults to "India" — change if you need other countries
        min_rating: optional floor, e.g. 4 to only show 4-star+ hotels
        limit: max results (free tier is fine with the default of 10)

    Returns:
        {"status": "ok", "hotels": [{"name", "rating", "city", "amenities", ...}]}
        or {"status": "error", "message": "..."} if the key is missing / call fails
    """
    if not HOTELS_API_KEY:
        return {
            "status": "error",
            "message": (
                "HOTELS_API_KEY is not set. Get a free key (no card) at "
                "https://hotels-api.com/register/ then export HOTELS_API_KEY=..."
            ),
        }

    params = {"city": city, "country": country, "limit": limit}
    if min_rating is not None:
        params["min_rating"] = min_rating

    try:
        resp = requests.get(
            f"{BASE_URL}/hotels/search",
            params=params,
            headers={"X-API-KEY": HOTELS_API_KEY},
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Request failed: {e}"}

    if not payload.get("success"):
        return {"status": "error", "message": payload.get("message") or "Unknown API error"}

    hotels = [
        {
            "hotel_id": h.get("id"),
            "name": h.get("name"),
            "city": h.get("city"),
            "rating": h.get("rating"),
            "amenities": h.get("amenities", []),
            "lat": h.get("lat"),
            "lng": h.get("lng"),
        }
        for h in payload.get("data", [])
    ]
    return {"status": "ok", "count": len(hotels), "hotels": hotels}


if __name__ == "__main__":
    city_arg = sys.argv[1] if len(sys.argv) > 1 else "Jaipur"
    min_rating_arg = None
    if "--min-rating" in sys.argv:
        idx = sys.argv.index("--min-rating")
        min_rating_arg = float(sys.argv[idx + 1])

    result = search_real_hotels(city_arg, min_rating=min_rating_arg)

    if result["status"] == "error":
        print("Error:", result["message"])
    else:
        print(f"Found {result['count']} hotels in {city_arg}:\n")
        for h in result["hotels"]:
            amenities = ", ".join(h["amenities"]) if h["amenities"] else "—"
            print(f"  {h['name']} — {'⭐' * int(h['rating'] or 0)} ({h['rating']}) | {amenities}")
