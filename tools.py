"""
tools.py — Person A's deliverable ("Mock World")

All "backend" state lives in three JSON files: flights.json, hotels.json,
calendar.json. No real database — this is intentional (see project plan,
section 1: "No database"). Each function below reads/writes those files
directly, so state persists across a demo run but stays dead simple.

Every function returns a plain Python dict, ready to be JSON-serialized
back to Claude as a tool_result.

RISKY_ACTIONS marks which tool names are irreversible / cost money —
that set is what agent.py's Risk Checker will look up.
"""

import json
import os
from datetime import datetime

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FLIGHTS_PATH = os.path.join(DATA_DIR, "flights.json")
HOTELS_PATH = os.path.join(DATA_DIR, "hotels.json")
CALENDAR_PATH = os.path.join(DATA_DIR, "calendar.json")

# Tool names that require human approval before running.
# This is the entire "Risk Checker" concept from the plan (section 2) —
# kept here so both A's tools and B's agent loop agree on the same list.
RISKY_ACTIONS = {"book_flight", "book_hotel", "make_payment"}


# ---------- low-level file helpers ----------

def _load(path):
    with open(path, "r") as f:
        return json.load(f)


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- read-only search tools (safe, no approval needed) ----------

def search_flights(origin: str, destination: str, date: str = None):
    """Find flights matching origin/destination, optionally filtered by date.

    Args:
        origin: 3-letter airport code, e.g. "BLR"
        destination: 3-letter airport code, e.g. "DEL"
        date: optional "YYYY-MM-DD" filter

    Returns:
        {"status": "ok", "flights": [...]} — only flights with seats_available > 0
    """
    flights = _load(FLIGHTS_PATH)
    results = [
        f for f in flights
        if f["origin"].upper() == origin.upper()
        and f["destination"].upper() == destination.upper()
        and f["seats_available"] > 0
        and (date is None or f["date"] == date)
    ]
    return {"status": "ok", "count": len(results), "flights": results}


def search_hotels(city: str, max_price_inr: float = None, min_rating: float = None):
    """Find hotels in a city, optionally filtered by price/rating.

    Args:
        city: 3-letter city/airport code, e.g. "DEL"
        max_price_inr: optional ceiling on price_per_night_inr
        min_rating: optional floor on rating

    Returns:
        {"status": "ok", "hotels": [...]} — only hotels with rooms_available > 0
    """
    hotels = _load(HOTELS_PATH)
    results = [
        h for h in hotels
        if h["city"].upper() == city.upper()
        and h["rooms_available"] > 0
        and (max_price_inr is None or h["price_per_night_inr"] <= max_price_inr)
        and (min_rating is None or h["rating"] >= min_rating)
    ]
    return {"status": "ok", "count": len(results), "hotels": results}


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


def book_flight(flight_id: str):
    """Book a flight — RISKY. Costs money, hard to reverse (refund w/ fees).
    agent.py's Risk Checker must intercept this and get human approval
    BEFORE it is ever called for real.

    Returns:
        {"status": "ok", "booking": {...}} or {"status": "error", "message": ...}
    """
    flights = _load(FLIGHTS_PATH)
    for f in flights:
        if f["flight_id"] == flight_id:
            if f["seats_available"] <= 0:
                return {"status": "error", "message": "No seats available"}
            f["seats_available"] -= 1
            _save(FLIGHTS_PATH, flights)
            booking = {
                "flight_id": flight_id,
                "airline": f["airline"],
                "price_inr": f["price_inr"],
                "booked_at": datetime.now().isoformat(timespec="seconds"),
            }
            return {"status": "ok", "booking": booking}
    return {"status": "error", "message": f"Flight {flight_id} not found"}


def book_hotel(hotel_id: str, nights: int):
    """Book a hotel — RISKY. Costs money, hard to reverse.

    Returns:
        {"status": "ok", "booking": {...}} or {"status": "error", "message": ...}
    """
    hotels = _load(HOTELS_PATH)
    for h in hotels:
        if h["hotel_id"] == hotel_id:
            if h["rooms_available"] <= 0:
                return {"status": "error", "message": "No rooms available"}
            h["rooms_available"] -= 1
            _save(HOTELS_PATH, hotels)
            booking = {
                "hotel_id": hotel_id,
                "name": h["name"],
                "nights": nights,
                "total_price_inr": h["price_per_night_inr"] * nights,
                "booked_at": datetime.now().isoformat(timespec="seconds"),
            }
            return {"status": "ok", "booking": booking}
    return {"status": "error", "message": f"Hotel {hotel_id} not found"}


# ---------- quick manual test ----------
if __name__ == "__main__":
    print("Flights BLR->DEL:", search_flights("BLR", "DEL"))
    print("Hotels in DEL under 4000:", search_hotels("DEL", max_price_inr=4000))
    print("Calendar on 2026-10-06:", check_calendar("2026-10-06"))
    print("Conflict check 11:30-12:30 on 2026-10-06:",
          find_conflicts("2026-10-06", "11:30", "12:30"))
