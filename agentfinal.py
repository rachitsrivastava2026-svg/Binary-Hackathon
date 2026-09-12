"""
agent.py — Person B's deliverable ("Claude Brain"), Gemini edition

Same job as always: wrap a tool-use loop around tools.py and bake the
Risk Checker (plan section 2) directly into it. This version calls
Google's Gemini API (real free tier, no card) via the official
`google-genai` SDK instead of Anthropic or Groq.

IMPORTANT: automatic function calling is explicitly disabled below
(automatic_function_calling=AutomaticFunctionCallingConfig(disable=True)).
Without that, the SDK would execute tool calls itself and our Risk
Checker would never get a chance to intercept a booking before it
happens — which is the entire point of this project.

Setup:
    pip install google-genai
    export GEMINI_API_KEY=AI...   (free key from aistudio.google.com/apikey)

app.py's interface to this file is UNCHANGED:
    agent = BoundaryAgent()
    result = agent.send("Plan my trip to Delhi")        # -> dict
    result = agent.resolve_approval(approved=True)      # -> dict
    agent.action_log                                     # -> list of dicts

result["type"] is one of: "final", "approval_required"
"""

import os
import json
from google import genai
from google.genai import types

import tools

# Gemini model IDs change fairly often (see ai.google.dev/gemini-api/docs/models).
# gemini-2.5-flash was retired for new users in 2026; gemini-3.6-flash is
# the current free-tier-eligible Flash model. If this one gets retired too,
# the 404 error message itself will tell you the replacement model ID.
MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """You are Boundary, a travel-planning assistant with real \
tools for searching flights/hotels, checking a calendar, and booking things.

Rules:
- Always use tools to look things up. Never invent flight numbers, prices, \
or calendar events — call search_flights / search_hotels / check_calendar.
- Before booking anything, make sure the trip fits the user's calendar. Use \
find_conflicts if there's any doubt.
- Calendar updates (update_calendar) are safe and reversible — just do them.
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
    "check_calendar": tools.check_calendar,
    "find_conflicts": tools.find_conflicts,
    "update_calendar": tools.update_calendar,
    "book_flight": tools.book_flight,
    "book_hotel": tools.book_hotel,
}

# ---- Tool schemas, Gemini FunctionDeclaration format --------------------
# Same 7 tools as always. Gemini's FunctionDeclaration takes a plain JSON
# Schema dict via parameters_json_schema — same shape you'd use for OpenAI,
# just a different field name to hang it on.
FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_flights",
        description="Search available flights between two airports, optionally on a specific date.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "3-letter airport code, e.g. BLR"},
                "destination": {"type": "string", "description": "3-letter airport code, e.g. DEL"},
                "date": {"type": "string", "description": "YYYY-MM-DD, optional"},
            },
            "required": ["origin", "destination"],
        },
    ),
    types.FunctionDeclaration(
        name="search_hotels",
        description="Search available hotels in a city, optionally filtered by max price or min rating.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "3-letter city code, e.g. DEL"},
                "max_price_inr": {"type": "number"},
                "min_rating": {"type": "number"},
            },
            "required": ["city"],
        },
    ),
    types.FunctionDeclaration(
        name="check_calendar",
        description="List calendar events, optionally filtered to one date (YYYY-MM-DD).",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "date": {"type": "string"},
            },
        },
    ),
    types.FunctionDeclaration(
        name="find_conflicts",
        description="Check whether a proposed time window overlaps any existing calendar event.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "date": {"type": "string"},
                "start_time": {"type": "string", "description": "HH:MM, 24h"},
                "end_time": {"type": "string", "description": "HH:MM, 24h"},
            },
            "required": ["date", "start_time", "end_time"],
        },
    ),
    types.FunctionDeclaration(
        name="update_calendar",
        description="Create or update a calendar event. Safe and reversible — no approval needed.",
        parameters_json_schema={
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
    ),
    types.FunctionDeclaration(
        name="book_flight",
        description="Book a flight by flight_id. IRREVERSIBLE and costs money — requires human approval.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "flight_id": {"type": "string"},
                "reasoning": {"type": "string", "description": "One sentence: what this books and why"},
                "confidence": {"type": "integer", "description": "0-100"},
            },
            "required": ["flight_id", "reasoning", "confidence"],
        },
    ),
    types.FunctionDeclaration(
        name="book_hotel",
        description="Book a hotel by hotel_id for a number of nights. IRREVERSIBLE and costs money — requires human approval.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "hotel_id": {"type": "string"},
                "nights": {"type": "integer"},
                "reasoning": {"type": "string", "description": "One sentence: what this books and why"},
                "confidence": {"type": "integer", "description": "0-100"},
            },
            "required": ["hotel_id", "nights", "reasoning", "confidence"],
        },
    ),
]

GEMINI_TOOLS = [types.Tool(function_declarations=FUNCTION_DECLARATIONS)]

GEMINI_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    tools=GEMINI_TOOLS,
    # This is the line that keeps the Risk Checker in charge — see the
    # module docstring above.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


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
    survives Streamlit reruns. Interface is identical to the Anthropic and
    Groq versions — only what's inside changed.

    Note: Gemini's `chat` object keeps conversation history internally
    (unlike the manual messages list we managed for Anthropic/Groq), so
    there's less bookkeeping here — but the pause-for-approval logic still
    needs somewhere to hold "results computed but not yet sent" for the
    (rare) case where a risky call shows up alongside safe ones in the
    same turn.
    """

    def __init__(self, model: str = MODEL):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.chat = self.client.chats.create(model=model, config=GEMINI_CONFIG)
        self.pending = None  # {"name": str, "args": dict} while awaiting approval
        self._safe_parts_waiting = []  # Parts computed but not yet sent this turn

        # History of every tool action, for app.py's dashboard counters and
        # itinerary panel. Each entry: {"tool", "event", "input", "result"}.
        # event is one of: "auto_allowed", "awaiting", "approved", "denied".
        self.action_log = []

    # ---- public API used by app.py (unchanged) ----

    def send(self, user_text: str) -> dict:
        response = self.chat.send_message(user_text)
        return self._handle_response(response)

    def resolve_approval(self, approved: bool) -> dict:
        if self.pending is None:
            raise RuntimeError("resolve_approval() called with no pending approval")

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

        parts = self._safe_parts_waiting + [types.Part.from_function_response(name=name, response=result)]
        self._safe_parts_waiting = []
        self.pending = None

        response = self.chat.send_message(parts)
        return self._handle_response(response)

    # ---- internals ----

    def _handle_response(self, response) -> dict:
        function_calls = response.function_calls or []

        if not function_calls:
            return {"type": "final", "text": (response.text or "").strip()}

        reply_text = (response.text or "").strip()
        parts = []

        for fc in function_calls:
            name = fc.name
            tool_input = dict(fc.args) if fc.args else {}

            if needs_approval(name):
                # Flush any safe results collected earlier in this same
                # turn before pausing, so nothing is lost.
                self._safe_parts_waiting = parts
                self.pending = {"name": name, "args": tool_input}
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
            parts.append(types.Part.from_function_response(name=name, response=result))

        # All calls this turn were safe — send results back and keep going
        # so the model can react to them.
        response = self.chat.send_message(parts)
        return self._handle_response(response)


# ---- quick manual smoke test (needs GEMINI_API_KEY set) ----
if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run this smoke test. Get a free key at aistudio.google.com/apikey")
    else:
        agent = BoundaryAgent()
        result = agent.send("Find me a flight from BLR to DEL on 2026-10-05.")
        print(json.dumps(result, indent=2))
