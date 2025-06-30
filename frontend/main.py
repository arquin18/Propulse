"""
Propulse Frontend Service
Main application entry point
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Propulse",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚀 Propulse")
st.subheader("AI-Powered Proposal Generation System")

# Main content
st.write("""
Welcome to Propulse! This system helps you generate high-quality proposals using AI.
Upload an RFP document or provide a description to get started.
""")

# File upload
uploaded_file = st.file_uploader("Upload RFP Document", type=["pdf", "docx"])

# Prompt input
prompt = st.text_area("Enter your proposal requirements:", height=200)

# Generate button
if st.button("Generate Proposal"):
    if not prompt and not uploaded_file:
        st.error("Please provide either a prompt or upload an RFP document.")
    else:
        with st.spinner("Generating proposal..."):
            # TODO: Implement proposal generation
            st.info("Proposal generation not implemented yet.") 