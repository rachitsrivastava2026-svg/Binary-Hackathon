"""
tools.py — Person A's deliverable ("Mock World"), now with a real data
option bolted on.

Mock JSON files (flights.json, calendar.json) are still the default
backend for flights and the calendar — nothing breaks if you run those
with no setup at all. Hotels are different: search_hotels / book_hotel
never touch a mock hotels.json file — they use Duffel (if DUFFEL_API_KEY
is set) or hotels-api.com's live hotel database (see hotel_lookup.py)
for everything.

If you set DUFFEL_API_KEY, search_flights() switches over to real
search results from Duffel (https://duffel.com) instead of the local
flights.json file, and search_hotels() switches to Duffel Stays instead
of hotels-api.com. Duffel is the current easiest self-serve option for
this kind of project: free signup, a real test mode (test tokens start
with "duffel_test_", no money ever moves), covers both flights and hotels
("Stays"). It replaced Amadeus's old free self-service tier, which
Amadeus shut down on July 17, 2026.

    pip install requests
    export DUFFEL_API_KEY=duffel_test_...   (free key from duffel.com)

book_flight / book_hotel no longer pretend to charge a card. Neither
MakeMyTrip nor Booking.com hand out open self-serve booking access to
individual developers — actually completing a purchase on either site
requires a commercial partner agreement and PCI-compliant payment
handling. So instead, once a booking is "approved" by the Risk Checker,
these functions decrement the local mock inventory (so the demo's state
stays consistent) and return a deep link to the real flight/hotel so a
human can finish paying on an actual site. That fits the project's
human-in-the-loop design anyway: the approval card becomes "here's what
I found, click through to actually book it" rather than "I just spent
your money."

Every function still returns a plain dict, JSON-serializable, same shape
as before. agent.py / app.py don't need to change at all.
"""

import json
import os
import urllib.parse
from datetime import datetime, timedelta

import requests  # pip install requests

from hotel_lookup import search_real_hotels as _search_real_hotels_api

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_PATH = os.path.join(DATA_DIR, "flights.json")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar.json")
# Note: there is no HOTELS_PATH — search_hotels/book_hotel never read or
# write a mock hotels.json file; they use Duffel or hotels-api.com only.

# Tool names that require human approval before running.
# This is the entire "Risk Checker" concept from the plan (section 2) —
# kept here so both A's tools and B's agent loop agree on the same list.
RISKY_ACTIONS = {"book_flight", "book_hotel", "make_payment"}


# ---------- external data source config ----------

# Set this env var to switch search_flights/search_hotels over to real
# Duffel results. Leave it unset and everything falls back to the mock
# JSON files, exactly like before.
DUFFEL_API_KEY = os.environ.get("DUFFEL_API_KEY")
DUFFEL_BASE_URL = "https://api.duffel.com"

# Rough, FIXED exchange rates, used only to show an approximate
# INR-equivalent price when Duffel returns an offer priced in another
# currency. These are not live rates — swap in a real FX API if you need
# accuracy for anything beyond a demo.
FX_TO_INR = {"INR": 1.0, "USD": 83.0, "GBP": 105.0, "EUR": 90.0}

# Duffel Stays searches by lat/long radius, not city code, so we need a
# lookup. Anchored on each city's commercial center (not the airport) so
# a normal-sized radius actually covers where hotels cluster. Add more
# cities here, or replace with a real geocoding call (Mapbox, Google,
# etc.) if you need more than a couple of fixed cities.
CITY_COORDS = {
    "DEL": {"latitude": 28.6304, "longitude": 77.2177, "name": "New Delhi"},  # Connaught Place
    "BLR": {"latitude": 12.9716, "longitude": 77.5946, "name": "Bengaluru"},  # MG Road / city center
}

# How wide to cast the net around that anchor point. Too small and a
# real search can come back with zero hotels even though the city is
# full of them.
STAYS_SEARCH_RADIUS_KM = 20

# In-memory cache of the last search results, keyed by id. This is what
# lets book_flight/book_hotel — which only receive an id, per the tool
# schema in agent.py — find the matching record. For flights that's mock
# JSON or a live Duffel search; for hotels it's always a live source
# (Duffel or hotels-api.com) — there's no mock hotel data to fall back to.
_FLIGHT_CACHE = {}
_HOTEL_CACHE = {}


def _duffel_headers(version="v2"):
    return {
        "Authorization": f"Bearer {DUFFEL_API_KEY}",
        "Duffel-Version": version,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


# ---------- low-level file helpers ----------

def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- deep links (used by book_flight / book_hotel) ----------

def _flight_deep_link(origin: str, destination: str, date: str) -> str:
    """Google Flights search link — stable, documented query format.
    Swap in a MakeMyTrip (or other OTA) link here if you have one you
    trust; MakeMyTrip's query-param format isn't publicly documented and
    changes often, so it's not safe to hardcode."""
    query = f"Flights from {origin.upper()} to {destination.upper()} on {date}"
    return "https://www.google.com/travel/flights?q=" + urllib.parse.quote(query)


def _hotel_deep_link(city_query: str, check_in: str, check_out: str,
                      adults: int = 1, rooms: int = 1) -> str:
    """Booking.com's public search URL — documented, stable query params."""
    params = {
        "ss": city_query,
        "checkin": check_in,
        "checkout": check_out,
        "group_adults": adults,
        "no_rooms": rooms,
    }
    return "https://www.booking.com/searchresults.html?" + urllib.parse.urlencode(params)


# ---------- read-only search tools (safe, no approval needed) ----------

def search_flights(origin: str, destination: str, date: str = None):
    """Find flights matching origin/destination, optionally filtered by date.

    Uses Duffel if DUFFEL_API_KEY is set, otherwise the local mock data.

    Args:
        origin: 3-letter airport code, e.g. "BLR"
        destination: 3-letter airport code, e.g. "DEL"
        date: optional "YYYY-MM-DD" filter

    Returns:
        {"status": "ok", "flights": [...]}
    """
    if DUFFEL_API_KEY:
        try:
            result = _search_flights_duffel(origin, destination, date)
            for f in result["flights"]:
                _FLIGHT_CACHE[f["flight_id"]] = f
            return result
        except Exception as e:
            print(f"[tools.py] Duffel flight search failed ({e}) — falling back to mock data")
    result = _search_flights_mock(origin, destination, date)
    for f in result["flights"]:
        _FLIGHT_CACHE[f["flight_id"]] = f
    return result


def _search_flights_mock(origin, destination, date):
    flights = _load(FLIGHTS_PATH)
    results = [
        f for f in flights
        if f["origin"].upper() == origin.upper()
        and f["destination"].upper() == destination.upper()
        and f["seats_available"] > 0
        and (date is None or f["date"] == date)
    ]
    return {"status": "ok", "count": len(results), "flights": results}


def _search_flights_duffel(origin, destination, date):
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
    resp = requests.post(
        f"{DUFFEL_BASE_URL}/air/offer_requests?return_offers=true",
        headers=_duffel_headers(),
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    offers = resp.json()["data"].get("offers", [])

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


def search_hotels(city: str, max_price_inr: float = None, min_rating: float = None):
    """Find hotels in a city, optionally filtered by price/rating.

    Uses Duffel Stays if DUFFEL_API_KEY is set and the city is in
    CITY_COORDS. Otherwise falls back to hotels-api.com's live hotel
    database (see hotel_lookup.py / search_real_hotels) — this tool no
    longer reads from the local mock hotels.json file at all.

    Note: Duffel Stays is currently limited to "closed user groups" —
    if your account doesn't have it enabled yet, this will fail over to
    hotels-api.com automatically (email stays@duffel.com to request access).

    Note: hotels-api.com doesn't return price, so max_price_inr can only
    be honored when Duffel is the actual source for this call — the
    hotels-api.com fallback ignores it and filters on min_rating only.

    Args:
        city: 3-letter city/airport code for Duffel, e.g. "DEL". If
            Duffel isn't used for this call, this is passed straight to
            hotels-api.com, which wants a full city name (e.g. "Jaipur")
            rather than a code.
        max_price_inr: optional ceiling on price_per_night_inr (Duffel only)
        min_rating: optional floor on rating

    Returns:
        {"status": "ok", "hotels": [...]}
    """
    if DUFFEL_API_KEY and city.upper() in CITY_COORDS:
        try:
            result = _search_hotels_duffel(city, max_price_inr, min_rating)
            if result["raw_count"] == 0:
                # Duffel answered but found nothing near the anchor point at
                # all (radius too tight, Stays not enabled on this account,
                # no inventory there yet) -- that's a real-source failure,
                # so fall back rather than tell the user "no hotels in Delhi".
                print(f"[tools.py] Duffel stays returned 0 raw results for {city} — falling back to hotels-api.com")
            else:
                for h in result["hotels"]:
                    _HOTEL_CACHE[h["hotel_id"]] = h
                return result
        except Exception as e:
            print(f"[tools.py] Duffel stays search failed ({e}) — falling back to hotels-api.com")

    # No mock JSON fallback — hotels-api.com is the real-data source for
    # every case Duffel doesn't cover. search_real_hotels() already caches
    # results into _HOTEL_CACHE for book_hotel to find later.
    return search_real_hotels(city, min_rating=min_rating)


def _search_hotels_duffel(city, max_price_inr, min_rating, nights=3):
    coords = CITY_COORDS[city.upper()]
    check_in = datetime.now().date() + timedelta(days=30)
    check_out = check_in + timedelta(days=nights)

    body = {
        "data": {
            "rooms": 1,
            "adults": 1,
            "check_in_date": check_in.isoformat(),
            "check_out_date": check_out.isoformat(),
            "location": {
                "radius": STAYS_SEARCH_RADIUS_KM,
                "geographic_coordinates": {
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"],
                },
            },
        }
    }
    resp = requests.post(
        f"{DUFFEL_BASE_URL}/stays/search",
        headers=_duffel_headers(version="v1"),
        json=body,
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json()["data"]["results"]

    hotels = []
    for r in results:
        acc = r["accommodation"]
        currency = acc.get("cheapest_rate_total_currency", "INR")
        total = float(acc.get("cheapest_rate_total_amount") or 0)
        price_per_night = round((total * FX_TO_INR.get(currency, 1.0)) / max(nights, 1))
        # review_score is out of 10 on Duffel; normalize to the 5-point
        # scale the rest of this project uses.
        rating = round((acc.get("review_score") or 0) / 2, 1)

        if max_price_inr is not None and price_per_night > max_price_inr:
            continue
        if min_rating is not None and rating < min_rating:
            continue

        hotels.append({
            "hotel_id": acc["id"],
            "name": acc["name"],
            "city": city.upper(),
            "rating": rating,
            "price_per_night_inr": price_per_night,
            "rooms_available": 1,  # Duffel confirms availability at booking time, not as a count here
            "amenities": [a.get("type", "") for a in acc.get("amenities", [])],
        })
    return {"status": "ok", "count": len(hotels), "raw_count": len(results), "hotels": hotels}


def check_calendar(date: str = None):
    """Return calendar events, optionally filtered to a single date.

    Args:
        date: optional "YYYY-MM-DD" filter

    Returns:
        {"status": "ok", "events": [...]}
    """
    events = _load(CALENDAR_PATH)
    if date:
        events = [e for e in events if e["date"] == date]
    events = sorted(events, key=lambda e: (e["date"], e["start_time"]))
    return {"status": "ok", "count": len(events), "events": events}


def find_conflicts(date: str, start_time: str, end_time: str):
    """Check whether a proposed time window overlaps any existing event.

    Used for the 'meeting moved' adaptation scene (demo act 3).

    Returns:
        {"status": "ok", "conflicts": [...]} — list of overlapping events
    """
    events = _load(CALENDAR_PATH)
    conflicts = []
    for e in events:
        if e["date"] != date:
            continue
        if start_time < e["end_time"] and end_time > e["start_time"]:
            conflicts.append(e)
    return {"status": "ok", "has_conflict": len(conflicts) > 0, "conflicts": conflicts}


def search_real_hotels(city: str, min_rating: float = None):
    """Look up REAL hotel names and star ratings in an Indian city via
    hotels-api.com's free tier (see hotel_lookup.py). This is genuine live
    data — not mock JSON.

    These results ARE bookable: each hotel is cached by hotel_id (same
    cache Duffel results use), so a later book_hotel(hotel_id=...) call
    finds it here. There's no mock hotels.json file for hotels at all —
    hotels-api.com doesn't return a live price, though, so the resulting
    booking's total_price_inr will be null — book_hotel hands back a
    deep link so the human can see/confirm the real price when finishing
    the booking themselves.

    Args:
        city: full city name, e.g. "Jaipur" (not a 3-letter code)
        min_rating: optional floor, e.g. 4 for 4-star and up

    Returns:
        {"status": "ok", "hotels": [{"hotel_id", "name", "rating", ...}]}
        or {"status": "error", "message": "..."} if HOTELS_API_KEY is missing
    """
    result = _search_real_hotels_api(city, country="India", min_rating=min_rating, limit=10)
    if result["status"] == "ok":
        for h in result["hotels"]:
            # hotels-api.com returns numeric ids; normalize to str so they
            # match the string hotel_id book_hotel's tool schema expects.
            hotel_id = str(h["hotel_id"])
            h["hotel_id"] = hotel_id
            _HOTEL_CACHE[hotel_id] = {
                "hotel_id": hotel_id,
                "name": h.get("name"),
                "city": h.get("city", city),
                "rating": h.get("rating"),
                "amenities": h.get("amenities", []),
                "price_per_night_inr": None,  # not provided by this source
                "rooms_available": None,      # not provided by this source
                "lat": h.get("lat"),
                "lng": h.get("lng"),
            }
    return result


# ---------- write tools (some are risky — see RISKY_ACTIONS) ----------

def update_calendar(event_id: str, title: str, date: str, start_time: str,
                     end_time: str, location: str = ""):
    """Create or update a calendar event. NOT in RISKY_ACTIONS — reversible,
    no money involved, so the plan treats this as auto-approved.

    Returns:
        {"status": "ok", "event": {...}, "action": "created"|"updated"}
    """
    events = _load(CALENDAR_PATH)
    for i, e in enumerate(events):
        if e["event_id"] == event_id:
            events[i] = {
                "event_id": event_id, "title": title, "date": date,
                "start_time": start_time, "end_time": end_time,
                "location": location,
            }
            _save(CALENDAR_PATH, events)
            return {"status": "ok", "action": "updated", "event": events[i]}

    new_event = {
        "event_id": event_id, "title": title, "date": date,
        "start_time": start_time, "end_time": end_time,
        "location": location,
    }
    events.append(new_event)
    _save(CALENDAR_PATH, events)
    return {"status": "ok", "action": "created", "event": new_event}


def book_flight(flight_id: str, reasoning: str = "", confidence: int = None):
    """Book a flight — RISKY. agent.py's Risk Checker must intercept this
    and get human approval BEFORE it is ever called.

    reasoning/confidence come from the model (see agent.py's tool schema,
    which requires them) and feed the "Why?" approval panel. They aren't
    needed for the booking logic itself, so they're just carried through
    into the returned record for the activity log / UI to show.

    Neither Duffel test mode nor a real OTA gets charged here. This
    decrements the local mock seat count (so the demo's state stays
    consistent across a run) and hands back a deep link to the real
    flight so a human finishes the purchase themselves.

    Returns:
        {"status": "ok", "booking": {...}} or {"status": "error", "message": ...}
    """
    cached = _FLIGHT_CACHE.get(flight_id)

    flights = _load(FLIGHTS_PATH)
    mock_match = next((f for f in flights if f["flight_id"] == flight_id), None)

    record = mock_match or cached
    if record is None:
        return {"status": "error", "message": f"Flight {flight_id} not found"}

    if mock_match is not None:
        if mock_match["seats_available"] <= 0:
            return {"status": "error", "message": "No seats available"}
        mock_match["seats_available"] -= 1
        _save(FLIGHTS_PATH, flights)

    deep_link = _flight_deep_link(record["origin"], record["destination"], record["date"])
    booking = {
        "flight_id": flight_id,
        "airline": record["airline"],
        "price_inr": record["price_inr"],
        "booked_at": datetime.now().isoformat(timespec="seconds"),
        "deep_link": deep_link,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    return {"status": "ok", "booking": booking, "deep_link": deep_link}


def book_hotel(hotel_id: str, nights: int, reasoning: str = "", confidence: int = None):
    """Book a hotel — RISKY. Same deep-link handoff as book_flight: no
    money moves here, the human finishes the booking on the real site.

    reasoning/confidence come from the model (see agent.py's tool schema,
    which requires them) and feed the "Why?" approval panel.

    Returns:
        {"status": "ok", "booking": {...}} or {"status": "error", "message": ...}
    """
    hotel_id = str(hotel_id)

    # No mock JSON fallback — book_hotel only ever books a hotel that was
    # already found by search_hotels or search_real_hotels, both of which
    # cache their results here (Duffel or hotels-api.com).
    record = _HOTEL_CACHE.get(hotel_id)
    if record is None:
        return {
            "status": "error",
            "message": f"Hotel {hotel_id} not found — search with search_hotels or search_real_hotels first",
        }

    # Duffel-sourced records carry a real rooms_available count; hotels-api.com
    # records set it to None since that source doesn't track availability.
    if isinstance(record.get("rooms_available"), int) and record["rooms_available"] <= 0:
        return {"status": "error", "message": "No rooms available"}

    price_per_night = record.get("price_per_night_inr")
    total_price_inr = price_per_night * nights if price_per_night is not None else None

    check_in = datetime.now().date() + timedelta(days=30)
    check_out = check_in + timedelta(days=nights)
    city_query = CITY_COORDS.get((record.get("city") or "").upper(), {}).get(
        "name", record.get("city") or record["name"]
    )
    deep_link = _hotel_deep_link(city_query, check_in.isoformat(), check_out.isoformat())

    booking = {
        "hotel_id": hotel_id,
        "name": record["name"],
        "nights": nights,
        "total_price_inr": total_price_inr,
        "price_note": (
            None if total_price_inr is not None
            else "This source doesn't provide live pricing — confirm the real price via the deep link before paying."
        ),
        "booked_at": datetime.now().isoformat(timespec="seconds"),
        "deep_link": deep_link,
        "reasoning": reasoning,
        "confidence": confidence,
    }
    return {"status": "ok", "booking": booking, "deep_link": deep_link}


# ---------- quick manual test ----------
if __name__ == "__main__":
    print("Flights BLR->DEL:", search_flights("BLR", "DEL"))
    print("Hotels in DEL under 4000:", search_hotels("DEL", max_price_inr=4000))
    print("Calendar on 2026-10-06:", check_calendar("2026-10-06"))
    print("Conflict check 11:30-12:30 on 2026-10-06:",
          find_conflicts("2026-10-06", "11:30", "12:30"))
    if DUFFEL_API_KEY:
        print("(Using live Duffel data for flights/hotels)")
    else:
        print("(DUFFEL_API_KEY not set — flights use mock JSON; hotels use hotels-api.com)")
