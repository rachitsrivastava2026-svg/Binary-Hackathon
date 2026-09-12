import streamlit as st
from agent4 import BoundaryAgent

st.set_page_config(page_title="Boundary — Guardrailed AI Agent", layout="wide")

# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = BoundaryAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent

# --- Top Dashboard (Feature B from Project Plan) ---
allowed = sum(1 for a in agent.action_log if a["event"] == "auto_allowed")
awaiting = sum(1 for a in agent.action_log if a["event"] == "awaiting")
approved = sum(1 for a in agent.action_log if a["event"] == "approved")
denied = sum(1 for a in agent.action_log if a["event"] == "denied")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🟢 Auto-Allowed", allowed)
col2.metric("🟡 Awaiting Approval", awaiting)
col3.metric("✅ Approved", approved)
col4.metric("🔴 Denied", denied)

st.divider()

# --- Main Layout ---
chat_col, side_col = st.columns([2, 1])

with chat_col:
    st.subheader("Chat Assistant")
    
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])
            if "card" in msg:
                card = msg["card"]
                st.warning(f"🟡 **APPROVAL REQUIRED: {card['tool_name']}**")
                st.write(f"Intent:")
                st.write(f"Risk Level: | Reversible:")
                st.write(f"Confidence:%")
                st.json(card["data_used"])

    # Handle Pending Approval Prompt
    if agent.pending:
        st.info("⚠️ Action paused waiting for your authorization.")
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("✅ Approve Action", type="primary"):
            res = agent.resolve_approval(approved=True)
            st.session_state.messages.append({"role": "assistant", "text": res["text"]})
            st.rerun()
        if btn_col2.button("❌ Deny Action"):
            res = agent.resolve_approval(approved=False)
            st.session_state.messages.append({"role": "assistant", "text": res["text"]})
            st.rerun()

    # User Input Chat Box
    elif user_input := st.chat_input("Plan my trip to Delhi..."):
        st.session_state.messages.append({"role": "user", "text": user_input})
        res = agent.send(user_input)
        
        msg_data = {"role": "assistant", "text": res.get("text", "")}
        if res["type"] == "approval_required":
            msg_data["card"] = res["card"]
        st.session_state.messages.append(msg_data)
        st.rerun()

# --- Side Panel (Action Log / Itinerary) ---
with side_col:
    st.subheader("🛡️ Real-Time Audit Log")
    for action in reversed(agent.action_log):
        with st.expander(f"{action['event'].upper()}: {action['tool']}"):
            st.write("**Input:**", action["input"])
            st.write("**Result:**", action["result"])
