"""
app.py — Person C's deliverable ("UI"). Streamlit front-end for Boundary.

Wires agent.py's BoundaryAgent (Groq / Llama 3.3, free tier) to a chat
interface, a dashboard of allow/pending/block counts, an itinerary panel,
and the "Why?" approval card whenever the agent hits a risky action.

Run locally:
    pip install streamlit groq python-dotenv
    streamlit run app.py

Deploy on Streamlit Cloud: push this file + agent.py + tools.py +
hotel_lookup.py + flights.json/calendar.json + requirements.txt to
GitHub (hotels.json is not needed — hotel search/booking is live-only
now, via Duffel or hotels-api.com), then add GROQ_API_KEY and
HOTELS_API_KEY under the app's Settings -> Secrets (see the block below
that bridges st.secrets into an environment variable agent.py can read).
"""
import os
import streamlit as st

# Bridge Streamlit Cloud's secrets manager into an env var, since agent.py
# reads os.environ.get("GROQ_API_KEY"). Locally this does nothing if you're
# already using a .env file — st.secrets just won't have anything to give.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "HOTELS_API_KEY" in st.secrets:
        os.environ["HOTELS_API_KEY"] = st.secrets["HOTELS_API_KEY"]
except Exception:
    pass  # no secrets.toml locally — fine, .env / exported var will cover it

from agent import BoundaryAgent

st.set_page_config(page_title="Boundary", page_icon="🧭", layout="wide")

# ---------- minimal theming, matches the pitch deck's palette ----------
st.markdown("""
<style>
.stApp { background: #FFFFFF; }
.boundary-header {
    background: #0B2B31; padding: 20px 24px; border-radius: 10px;
    margin-bottom: 20px;
}
.boundary-header h1 { color: #FFFFFF; margin: 0; font-size: 26px; }
.boundary-header p { color: #02C39A; margin: 4px 0 0; font-style: italic; font-size: 13px; }
.chip { border-radius: 10px; padding: 12px; text-align: left; }
.chip .n { font-size: 26px; font-weight: bold; display: block; }
.chip .l { font-size: 11px; font-weight: bold; letter-spacing: 0.5px; }
.approval-box {
    border: 1px solid #D9E2E4; border-radius: 10px; overflow: hidden;
    margin: 10px 0;
}
.approval-head { background: #C9820A; color: #FFFFFF; padding: 10px 16px; font-weight: bold; }
.approval-body { padding: 12px 16px; background: #FFFFFF; }
.approval-row { display: flex; font-size: 13px; padding: 4px 0; }
.approval-row .k { width: 110px; color: #6B7280; font-weight: bold; }
.approval-row .v { color: #1F2933; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="boundary-header">
  <h1>BOUNDARY</h1>
  <p>An AI agent with a conscience — free, open-model backend (Groq / Llama 3.3)</p>
</div>
""", unsafe_allow_html=True)

# ---------- session state ----------
if "agent" not in st.session_state:
    try:
        st.session_state.agent = BoundaryAgent()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # list of {"role": "user"/"agent", "text": ...}

if "pending_card" not in st.session_state:
    st.session_state.pending_card = None  # holds the approval card dict while waiting

agent = st.session_state.agent

# ---------- layout: chat (left) + dashboard (right) ----------
col_chat, col_side = st.columns([1.4, 1])

with col_side:
    st.subheader("Dashboard")
    counts = {"auto_allowed": 0, "approved": 0, "awaiting": 0, "denied": 0}
    for entry in agent.action_log:
        counts[entry["event"]] = counts.get(entry["event"], 0) + 1
    allowed = counts["auto_allowed"] + counts["approved"]
    pending = 1 if st.session_state.pending_card else 0
    blocked = counts["denied"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="chip" style="background:#E7F6EF;"><span class="n" style="color:#1E7B4D;">{allowed}</span><span class="l" style="color:#1E7B4D;">ALLOWED</span></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="chip" style="background:#FBF0DD;"><span class="n" style="color:#C9820A;">{pending}</span><span class="l" style="color:#C9820A;">PENDING</span></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="chip" style="background:#FBE9EB;"><span class="n" style="color:#B23A48;">{blocked}</span><span class="l" style="color:#B23A48;">BLOCKED</span></div>', unsafe_allow_html=True)

    st.subheader("Activity Log")
    if not agent.action_log:
        st.caption("Nothing yet.")
    else:
        for entry in reversed(agent.action_log[-10:]):
            st.caption(f"**{entry['tool']}** — {entry['event']}")

with col_chat:
    # replay chat history
    for msg in st.session_state.chat_log:
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.write(msg["text"])

    # show a pending approval card, if any
    if st.session_state.pending_card:
        card = st.session_state.pending_card
        st.markdown(f"""
        <div class="approval-box">
          <div class="approval-head">⚠️ APPROVAL REQUIRED — {card['tool_name']}</div>
          <div class="approval-body">
            <div class="approval-row"><span class="k">Intent</span><span class="v">{card['intent']}</span></div>
            <div class="approval-row"><span class="k">Risk</span><span class="v">{card['risk']}</span></div>
            <div class="approval-row"><span class="k">Data used</span><span class="v">{card['data_used']}</span></div>
            <div class="approval-row"><span class="k">Reversible</span><span class="v">{card['reversible']}</span></div>
            <div class="approval-row"><span class="k">Confidence</span><span class="v">{card['confidence']}%</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        b1, b2 = st.columns(2)
        approve_clicked = b1.button("✅ Allow", use_container_width=True)
        deny_clicked = b2.button("🚫 Deny", use_container_width=True)

        if approve_clicked or deny_clicked:
            st.session_state.pending_card = None
            with st.spinner("Working…"):
                result = agent.resolve_approval(approved=approve_clicked)
            if result["type"] == "approval_required":
                if result["text"]:
                    st.session_state.chat_log.append({"role": "agent", "text": result["text"]})
                st.session_state.pending_card = result["card"]
            else:
                st.session_state.chat_log.append({"role": "agent", "text": result["text"]})
            st.rerun()

    # chat input — disabled while an approval is pending, to keep the flow clean
    user_text = st.chat_input(
        "Type a request, e.g. 'plan my trip to Delhi and book everything'…",
        disabled=bool(st.session_state.pending_card),
    )

    if user_text:
        st.session_state.chat_log.append({"role": "user", "text": user_text})
        with st.spinner("Thinking…"):
            result = agent.send(user_text)
        if result["type"] == "approval_required":
            if result["text"]:
                st.session_state.chat_log.append({"role": "agent", "text": result["text"]})
            st.session_state.pending_card = result["card"]
        else:
            st.session_state.chat_log.append({"role": "agent", "text": result["text"]})
        st.rerun()
