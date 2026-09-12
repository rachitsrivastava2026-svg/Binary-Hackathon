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
import time
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
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


class BoundaryAgent:
    def __init__(self):
        # Fallback handling to ensure API key is collected seamlessly
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()
            
        self.chat = self.client.chats.create(model=MODEL, config=GEMINI_CONFIG)
        self.action_log = []
        self.pending_tool_call = None

    def _send_with_retry(self, message_content):
        """
        Helper method to communicate with Gemini API.
        Implements short programmatic pacing pauses and resilient 
        exponential backoff to gracefully absorb 429 Rate Limits on the Free Tier.
        """
        # Enforce an initial base spacing delay to respect the free-tier rate limitations (15 RPM)
        time.sleep(4.5)
        
        retries = 5
        delay = 4.0
        for attempt in range(retries):
            try:
                response = self.chat.send_message(message_content)
                return response
            except Exception as e:
                err_msg = str(e)
                # Intercept 429 Rate Limit / Resource Exhausted errors
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    if attempt < retries - 1:
                        print(f"[BoundaryAgent] Rate limit (429) encountered. Retrying in {delay}s...")
                        time.sleep(delay)
                        delay *= 2  # Exponential backoff escalation
                        continue
                raise e

    def send(self, user_text: str) -> dict:
        """
        Accepts raw input string from the user, updates chat context,
        and cycles through the manual tool-execution loop.
        """
        current_input = user_text
        
        while True:
            response = self._send_with_retry(current_input)
            
            # Check if Gemini wants to call a tool
            if response.function_calls:
                tool_call = response.function_calls[0]
                tool_name = tool_call.name
                tool_args = tool_call.args
                
                # Check Risk Profile
                if needs_approval(tool_name):
                    self.pending_tool_call = tool_call
                    return {
                        "type": "approval_required",
                        "message": f"I need your explicit confirmation to proceed with booking.",
                        "tool_name": tool_name,
                        "arguments": tool_args,
                        "risk_metadata": RISK_METADATA.get(tool_name, {"risk": "Unknown", "reversible": "Unknown"})
                    }
                
                # Safe internal execution (e.g., search or calendar read/writes)
                if tool_name in TOOL_FUNCTIONS:
                    try:
                        result = TOOL_FUNCTIONS[tool_name](**tool_args)
                    except Exception as e:
                        result = {"error": str(e)}
                else:
                    result = {"error": f"Tool '{tool_name}' is not recognized."}
                
                # Log execution tracking
                self.action_log.append({
                    "tool": tool_name,
                    "arguments": tool_args,
                    "status": "executed",
                    "result": result
                })
                
                # Feed tool results back to Gemini context for the next decision phase
                current_input = types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=tool_name,
                            response={"result": result}
                        )
                    ]
                )
                continue
                
            # No tool execution requested; Return final user response string
            return {
                "type": "final",
                "message": response.text
            }

    def resolve_approval(self, approved: bool) -> dict:
        """
        Handles user decisions regarding a blocked pending risky tool.
        """
        if not self.pending_tool_call:
            return {"type": "final", "message": "No pending transactions found requiring action."}
            
        tool_call = self.pending_tool_call
        tool_name = tool_call.name
        tool_args = tool_call.args
        self.pending_tool_call = None
        
        if approved:
            if tool_name in TOOL_FUNCTIONS:
                try:
                    result = TOOL_FUNCTIONS[tool_name](**tool_args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Tool '{tool_name}' is not recognized."}
                
            self.action_log.append({
                "tool": tool_name,
                "arguments": tool_args,
                "status": "approved_and_executed",
                "result": result
            })
            
            # Send confirmation response payload back into agent text generation thread
            next_input = types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result}
                    )
                ]
            )
        else:
            self.action_log.append({
                "tool": tool_name,
                "arguments": tool_args,
                "status": "rejected_by_user",
                "result": "User canceled transaction."
            })
            next_input = f"The user rejected your proposal to call '{tool_name}'. Please modify your plan or suggest alternative actions."
            
        # Re-enter the event-processing loop with the outcome context
        return self.send(next_input)
