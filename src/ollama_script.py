#!/usr/bin/env python3
"""
Streamlit app to browse and preview Figma components or frames,
and then generate React components and CSS files using Ollama server
in a multi-step workflow with iterative feedback.

Usage:
    export FIGMA_TOKEN="your_figma_token"
    export FIGMA_FILE_KEY="your_figma_file_key"
    export OLLAMA_MODEL="llama2:latest"  # or your preferred Ollama model
    pip install streamlit requests nest-asyncio python-dotenv
    ollama serve      # ensure Ollama server is running by default on localhost:11434
    streamlit run figma_streamlit_app.py
"""

import os
import json
import requests
import nest_asyncio
import streamlit as st
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

# Apply nest_asyncio for Streamlit async compatibility
nest_asyncio.apply()

FIGMA_API_BASE_URL = "https://api.figma.com/v1"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2-vision:11b")


def call_ollama(system_prompt: str, user_prompt: str) -> str:
    """
    Send a chat request to Ollama server and return the assistant content.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    resp = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    # Ollama format: {"message": {"role":...,"content":...}}
    return data.get("message", {}).get("content", "")


def get_env_var(name: str) -> str:
    val = os.getenv(name)
    if not val:
        st.error(f"Environment variable '{name}' is not set.")
        st.stop()
    return val


@st.cache_data(show_spinner=False)
def fetch_components(token: str, file_key: str) -> List[Dict]:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}/components"
    resp = requests.get(url, headers={"X-Figma-Token": token}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("meta", {}).get("components", [])


@st.cache_data(show_spinner=False)
def fetch_frames(token: str, file_key: str) -> List[Dict]:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}"
    resp = requests.get(url, headers={"X-Figma-Token": token}, timeout=10)
    resp.raise_for_status()
    document = resp.json().get("document", {})
    frames: List[Dict] = []

    def recurse(node: Dict):
        if node.get("type") == "FRAME":
            frames.append({"name": node["name"], "node_id": node["id"]})
        for child in node.get("children", []):
            recurse(child)

    recurse(document)
    return frames


@st.cache_data(show_spinner=False)
def fetch_image_url(
    token: str, file_key: str, node_id: str, fmt: str = "png"
) -> Optional[str]:
    url = f"{FIGMA_API_BASE_URL}/images/{file_key}"
    resp = requests.get(
        url,
        headers={"X-Figma-Token": token},
        params={"ids": node_id, "format": fmt},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("images", {}).get(node_id)


@st.cache_data(show_spinner=False)
def fetch_node_details(token: str, file_key: str, node_id: str) -> Dict:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}/nodes"
    resp = requests.get(
        url, headers={"X-Figma-Token": token}, params={"ids": node_id}, timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("nodes", {}).get(node_id, {}).get("document", {})


def main():
    # Page setup
    st.set_page_config(page_title="Figma to React via Ollama", layout="wide")
    st.title("Figma Component & CSS Generator (Ollama)")

    # Credentials
    token = get_env_var("FIGMA_TOKEN")
    file_key = get_env_var("FIGMA_FILE_KEY")

    # Fetch items
    comps = fetch_components(token, file_key)
    items = comps if comps else fetch_frames(token, file_key)
    label = "Components" if comps else "Frames"
    if not items:
        st.error("No items found in the specified Figma file.")
        return

    st.sidebar.header(label)
    mapping = {item["name"]: item for item in items}
    selected = st.sidebar.multiselect(f"Select {label}", options=sorted(mapping.keys()))
    if not selected:
        st.sidebar.info(f"Choose one or more {label.lower()} to preview.")
        return

    # Prepare data and preview
    session = st.session_state
    if "results" not in session:
        session["results"] = {}

    component_data = []
    for name in selected:
        node_id = mapping[name]["node_id"]
        image_url = fetch_image_url(token, file_key, node_id)
        details = fetch_node_details(token, file_key, node_id)
        st.subheader(name)
        if image_url:
            st.image(image_url, use_column_width=True)
        with st.expander("Raw Figma JSON"):
            st.json(details)
        component_data.append({"name": name, "json": details, "image_url": image_url})

    # Generation trigger
    if st.button("Create React & CSS Files via Ollama"):
        for comp in component_data:
            name = comp["name"]
            basename = name.replace(" ", "")
            # Step 1: JSX skeleton
            sys1 = "Generate a React JSX file named COMPONENT.jsx with only JSX skeleton, no styles, return code only."
            user1 = json.dumps(comp["json"])
            jsx = call_ollama(sys1, user1)

            # Step 2: CSS file
            sys2 = "Generate a CSS file named COMPONENT.css with class selectors matching the JSX, return code only."
            user2 = json.dumps(comp["json"])
            css = call_ollama(sys2, user2)

            # Step 3: Combine
            sys3 = "Combine JSX and CSS into COMPONENT.jsx, include `import './COMPONENT.css'`, return code only."
            user3 = f"JSX:\n{jsx}\nCSS:\n{css}\nImageURL: {comp['image_url']}"
            final = call_ollama(sys3, user3)

            session["results"][name] = {"jsx": final, "css": css}

    # Display generated files
    for name, files in st.session_state.get("results", {}).items():
        basename = name.replace(" ", "")
        st.markdown(f"### {name}")
        st.markdown(f"**{basename}.jsx**")
        st.code(files["jsx"], language="javascript")
        st.markdown(f"**{basename}.css**")
        st.code(files["css"], language="css")

    # Feedback loop
    feedback = st.text_area("Feedback (refine the code):")
    if st.button("Submit Feedback") and feedback:
        for comp in component_data:
            name = comp["name"]
            prev = session["results"].get(name, {})
            sys_fb = "Revise COMPONENT.jsx and COMPONENT.css based on user feedback: RETURN updated code files only."
            user_fb = json.dumps(
                {
                    "jsx": prev.get("jsx", ""),
                    "css": prev.get("css", ""),
                    "json": comp["json"],
                    "image_url": comp["image_url"],
                    "feedback": feedback,
                }
            )
            updated = call_ollama(sys_fb, user_fb)
            # Expect updated contains two code blocks separated
            parts = updated.split("\n---\n")
            if len(parts) >= 2:
                session["results"][name]["jsx"] = parts[0]
                session["results"][name]["css"] = parts[1]


if __name__ == "__main__":
    main()
