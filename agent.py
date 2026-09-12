"""
agent.py — Person B's deliverable ("Claude Brain"), Groq / open-source edition

Same job as always: wrap a tool-use loop around tools.py and bake the
Risk Checker (plan section 2) directly into it. This version calls Groq's
free API, which runs open-weight models (Llama 3.3 70B) instead of a
closed proprietary model — genuinely free, no credit card, no billing
surprises mid-demo. Groq's API is OpenAI-compatible, so this uses the
standard "tools" / "tool_calls" chat-completions shape.

Setup:
    pip install groq python-dotenv
    export GROQ_API_KEY=gsk_...   (free key from console.groq.com/keys — no card)

Put GROQ_API_KEY in a local .env file (never commit it, never paste the
literal key into source code — pass the *name* of the env var, not the
key itself, to os.environ.get()).

app.py's interface to this file is UNCHANGED:
    agent = BoundaryAgent()
    result = agent.send("Plan my trip to Delhi")        # -> dict
    result = agent.resolve_approval(approved=True)      # -> dict
    agent.action_log                                     # -> list of dicts

result["type"] is one of: "final", "approval_required"
"""
from dotenv import load_dotenv
load_dotenv()

import os
import json
from groq import Groq

import tools

# Groq's free tier: ~14,400 requests/day, 30/minute, no card required.
# llama-3.3-70b-versatile is an open-weight model (Meta's Llama 3.3),
# served on Groq's fast inference hardware. If this model is ever
# retired, Groq's error message will name the replacement.
MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """You are Boundary, a travel-planning assistant with real \
tools for searching flights/hotels, checking a calendar, and booking things.

Rules:
- Always use tools to look things up. Never invent flight numbers, prices, \
or calendar events — call search_flights / search_hotels / check_calendar.
- Before booking anything, make sure the trip fits the user's calendar. Use \
find_conflicts if there's any doubt.
- Calendar updates (update_calendar) are safe and reversible — just do them.
- For hotels, search_hotels is the default tool. It uses Duffel if \
DUFFEL_API_KEY is set for a Duffel-covered city (pass a 3-letter code \
like "DEL"), otherwise it uses hotels-api.com's live hotel database — \
there is no mock hotel data anymore, so for that fallback pass a full \
city name (e.g. "Jaipur") rather than a code. Its hotel_id IS bookable \
with book_hotel either way. hotels-api.com doesn't return a live price, \
so a booking sourced from it will come back with total_price_inr = null \
— tell the user to confirm the real price via the deep link. Only call \
search_real_hotels directly if you specifically want hotels-api.com's \
data even when Duffel would otherwise be used for that city.
- Booking a flight or hotel is NOT reversible without fees. Whenever you \
call book_flight or book_hotel, you must also fill in the tool's \
`reasoning` argument (one plain sentence: what you're doing and why) and a \
`confidence` argument (0-100, how sure you are this is the right choice \
based on what the user asked for).
- Call one tool at a time rather than several at once, especially before a \
booking — it keeps the human approval flow clean.
- After a tool result comes back, briefly tell the user what happened in \
plain, friendly language. Don't dump raw JSON at them.
"""

# ---- Risk Checker ----------------------------------------------------
RISKY_ACTIONS = tools.RISKY_ACTIONS  # {"book_flight", "book_hotel", "make_payment"}


def needs_approval(tool_name: str) -> bool:
    return tool_name in RISKY_ACTIONS


RISK_METADATA = {
    "book_flight": {
        "risk": "Medium",
        "reversible": "No — refund only, with fees",
    },
    "book_hotel": {
        "risk": "Medium",
        "reversible": "No — refund/cancellation policy dependent",
    },
    "make_payment": {
        "risk": "High",
        "reversible": "No — irreversible once processed",
    },
}

TOOL_FUNCTIONS = {
    "search_flights": tools.search_flights,
    "search_hotels": tools.search_hotels,
    "search_real_hotels": tools.search_real_hotels,
    "check_calendar": tools.check_calendar,
    "find_conflicts": tools.find_conflicts,
    "update_calendar": tools.update_calendar,
    "book_flight": tools.book_flight,
    "book_hotel": tools.book_hotel,
}

# ---- Tool schemas, OpenAI/Groq function-calling format -----------------
# Groq's chat.completions API is OpenAI-compatible: each tool is
# {"type": "function", "function": {name, description, parameters}}.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two airports, optionally on a specific date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "3-letter airport code, e.g. BLR"},
                    "destination": {"type": "string", "description": "3-letter airport code, e.g. DEL"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, optional"},
                },
                "required": ["origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search real hotels in a city (Duffel where covered, otherwise hotels-api.com's live database — no mock data). Bookable hotel_id either way. max_price_inr only has effect on Duffel results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "3-letter code (e.g. DEL) for a Duffel-covered city, otherwise a full city name (e.g. 'Jaipur') for the hotels-api.com fallback"},
                    "max_price_inr": {"type": "number"},
                    "min_rating": {"type": "number"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_real_hotels",
            "description": "Look up REAL hotel names and star ratings in an Indian city using live data from hotels-api.com. Its hotel_id IS bookable via book_hotel (no live price, though — use search_hotels instead if you need a known price up front).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Full city name, e.g. 'Jaipur' (not a 3-letter code)"},
                    "min_rating": {"type": "number", "description": "Optional floor, e.g. 4 for 4-star and up"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_calendar",
            "description": "List calendar events, optionally filtered to one date (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {"date": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_conflicts",
            "description": "Check whether a proposed time window overlaps any existing calendar event.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "start_time": {"type": "string", "description": "HH:MM, 24h"},
                    "end_time": {"type": "string", "description": "HH:MM, 24h"},
                },
                "required": ["date", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_calendar",
            "description": "Create or update a calendar event. Safe and reversible — no approval needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string", "description": "Existing id to update, or a new unique id to create"},
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["event_id", "title", "date", "start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight by flight_id. IRREVERSIBLE and costs money — requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "reasoning": {"type": "string", "description": "One sentence: what this books and why"},
                    "confidence": {"type": "integer", "description": "0-100"},
                },
                "required": ["flight_id", "reasoning", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_hotel",
            "description": "Book a hotel by hotel_id for a number of nights. IRREVERSIBLE and costs money — requires human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hotel_id": {"type": "string"},
                    "nights": {"type": "integer"},
                    "reasoning": {"type": "string", "description": "One sentence: what this books and why"},
                    "confidence": {"type": "integer", "description": "0-100"},
                },
                "required": ["hotel_id", "nights", "reasoning", "confidence"],
            },
        },
    },
]


def _execute_tool(name: str, tool_input: dict) -> dict:
    """Call the matching function in tools.py, stripping the Why-panel-only
    fields (reasoning/confidence) that aren't real function arguments."""
    clean_input = {k: v for k, v in tool_input.items() if k not in ("reasoning", "confidence")}
    fn = TOOL_FUNCTIONS[name]
    return fn(**clean_input)


def _build_approval_card(tool_name: str, tool_input: dict) -> dict:
    """Assemble the data app.py needs to render the 'Why panel' approval card."""
    meta = RISK_METADATA.get(tool_name, {"risk": "Unknown", "reversible": "Unknown"})
    data_used = {k: v for k, v in tool_input.items() if k not in ("reasoning", "confidence")}
    return {
        "tool_name": tool_name,
        "intent": tool_input.get("reasoning", f"Requesting to run {tool_name}"),
        "risk": meta["risk"],
        "data_used": data_used,
        "reversible": meta["reversible"],
        "confidence": tool_input.get("confidence", 0),
    }


class BoundaryAgent:
    """One instance = one conversation. Put it in st.session_state so it
    survives Streamlit reruns. Interface is identical to the Gemini/
    Anthropic versions — only what's inside changed.

    Unlike Gemini's `chat` object, Groq's OpenAI-compatible API doesn't
    keep history for you — this class manages the full `messages` list
    itself, the same way you would with the raw OpenAI SDK.
    """

    def __init__(self, model: str = MODEL):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key (no card) at "
                "console.groq.com/keys, then either export it or put it "
                "in a local .env file as GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.pending = None  # {"id": str, "name": str, "args": dict} while awaiting approval
        self._safe_tool_messages_waiting = []  # tool-result messages computed but not yet sent this turn

        # History of every tool action, for app.py's dashboard counters and
        # itinerary panel. Each entry: {"tool", "event", "input", "result"}.
        # event is one of: "auto_allowed", "awaiting", "approved", "denied".
        self.action_log = []

    # ---- public API used by app.py (unchanged) ----

    def send(self, user_text: str) -> dict:
        self.messages.append({"role": "user", "content": user_text})
        response = self._call_model()
        return self._handle_response(response)

    def resolve_approval(self, approved: bool) -> dict:
        if self.pending is None:
            raise RuntimeError("resolve_approval() called with no pending approval")

        call_id = self.pending["id"]
        name = self.pending["name"]
        tool_input = self.pending["args"]

        if approved:
            result = _execute_tool(name, tool_input)
        else:
            result = {"status": "denied", "message": "User denied this action."}

        self.action_log.append({
            "tool": name,
            "event": "approved" if approved else "denied",
            "input": tool_input,
            "result": result,
        })

        tool_messages = self._safe_tool_messages_waiting + [{
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result),
        }]
        self._safe_tool_messages_waiting = []
        self.pending = None
        self.messages.extend(tool_messages)

        response = self._call_model()
        return self._handle_response(response)

    # ---- internals ----

    def _call_model(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,  # lower = more reliable tool calling
        )

    def _handle_response(self, response) -> dict:
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            self.messages.append({"role": "assistant", "content": message.content or ""})
            return {"type": "final", "text": (message.content or "").strip()}

        # Record the assistant's tool-call turn in history before anything else.
        self.messages.append(message)

        reply_text = (message.content or "").strip()
        tool_messages = []

        for tc in tool_calls:
            name = tc.function.name
            tool_input = json.loads(tc.function.arguments) if tc.function.arguments else {}

            if needs_approval(name):
                # Flush any safe results collected earlier in this same
                # turn before pausing, so nothing is lost.
                self._safe_tool_messages_waiting = tool_messages
                self.pending = {"id": tc.id, "name": name, "args": tool_input}
                self.action_log.append({
                    "tool": name,
                    "event": "awaiting",
                    "input": tool_input,
                    "result": None,
                })
                return {
                    "type": "approval_required",
                    "text": reply_text,
                    "card": _build_approval_card(name, tool_input),
                }

            result = _execute_tool(name, tool_input)
            self.action_log.append({
                "tool": name,
                "event": "auto_allowed",
                "input": tool_input,
                "result": result,
            })
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        # All calls this turn were safe — send results back and keep going
        # so the model can react to them.
        self.messages.extend(tool_messages)
        response = self._call_model()
        return self._handle_response(response)


# ---- quick manual smoke test (needs GROQ_API_KEY set) ----
if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Set GROQ_API_KEY to run this smoke test. Get a free key (no card) at console.groq.com/keys")
    else:
        agent = BoundaryAgent()
        result = agent.send("Find me a flight from BLR to DEL on 2026-10-05.")
        print(json.dumps(result, indent=2))
