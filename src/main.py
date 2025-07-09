#!/usr/bin/env python3
"""
Streamlit app to browse and preview Figma components or frames,
and then generate React components and CSS files using the OpenAI Agents SDK in a multi-agent workflow,
with an iterative feedback loop for refining code.

Usage:
    export FIGMA_TOKEN="your_figma_token"
    export FIGMA_FILE_KEY="your_figma_file_key"
    export OPENAI_API_KEY="your_openai_api_key"
    pip install streamlit requests nest-asyncio openai-agents python-dotenv
    streamlit run figma_streamlit_app.py
"""

import os
import json
import requests
import nest_asyncio
import streamlit as st
from typing import Dict, List, Optional
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv

load_dotenv()

# Allow async operations inside Streamlit
nest_asyncio.apply()

FIGMA_API_BASE_URL = "https://api.figma.com/v1"


def get_env_var(name: str) -> str:
    val = os.getenv(name)
    if not val:
        st.error(f"Environment variable '{name}' is not set.")
        st.stop()
    return val


@st.cache_data(show_spinner=False)
def fetch_components(token: str, file_key: str) -> List[Dict]:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}/components"
    r = requests.get(url, headers={"X-Figma-Token": token}, timeout=10)
    r.raise_for_status()
    return r.json().get("meta", {}).get("components", [])


@st.cache_data(show_spinner=False)
def fetch_frames(token: str, file_key: str) -> List[Dict]:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}"
    r = requests.get(url, headers={"X-Figma-Token": token}, timeout=10)
    r.raise_for_status()
    doc = r.json().get("document", {})
    frames: List[Dict] = []

    def traverse(node: Dict):
        if node.get("type") == "FRAME":
            frames.append({"name": node["name"], "node_id": node["id"]})
        for child in node.get("children", []):
            traverse(child)

    traverse(doc)
    return frames


@st.cache_data(show_spinner=False)
def fetch_image_url(
    token: str, file_key: str, node_id: str, fmt: str = "png"
) -> Optional[str]:
    url = f"{FIGMA_API_BASE_URL}/images/{file_key}"
    r = requests.get(
        url,
        headers={"X-Figma-Token": token},
        params={"ids": node_id, "format": fmt},
        timeout=10,
    )
    r.raise_for_status()
    return r.json().get("images", {}).get(node_id)


@st.cache_data(show_spinner=False)
def fetch_node_details(token: str, file_key: str, node_id: str) -> Dict:
    url = f"{FIGMA_API_BASE_URL}/files/{file_key}/nodes"
    r = requests.get(
        url, headers={"X-Figma-Token": token}, params={"ids": node_id}, timeout=10
    )
    r.raise_for_status()
    return r.json().get("nodes", {}).get(node_id, {}).get("document", {})


@function_tool
def dummy_tool(input_data: str) -> str:
    return input_data


def main():
    st.set_page_config(page_title="Figma to React Explorer", layout="wide")
    st.title("Figma to React Component & CSS Generator")

    token = get_env_var("FIGMA_TOKEN")
    file_key = get_env_var("FIGMA_FILE_KEY")
    _ = get_env_var("OPENAI_API_KEY")

    comps = fetch_components(token, file_key)
    items = comps if comps else fetch_frames(token, file_key)
    label = "Components" if comps else "Frames"
    if not items:
        st.error("No items found in the Figma file.")
        return

    st.sidebar.header(label)
    name_map = {item["name"]: item for item in items}
    selected = st.sidebar.multiselect(
        f"Select {label}", options=sorted(name_map.keys())
    )
    if not selected:
        st.sidebar.info(
            f"Select one or more {label.lower()} to preview and prepare generation."
        )
        return

    # Prepare component data
    component_infos = []
    for name in selected:
        item = name_map[name]
        node_id = item["node_id"]
        st.subheader(name)
        img_url = fetch_image_url(token, file_key, node_id)
        details = fetch_node_details(token, file_key, node_id)
        component_infos.append({"name": name, "json": details, "image_url": img_url})
        if img_url:
            st.image(img_url, use_column_width=True)
        with st.expander("Component JSON data"):
            st.json(details)

    # Initialize session state
    if "generation_results" not in st.session_state:
        st.session_state.generation_results = {}

    # Generate code on demand
    if st.button("Create React Component & CSS Files"):
        for info in component_infos:
            key = info["name"]
            struct = Agent(
                name="structure_agent",
                instructions=(
                    "Given the Figma JSON and image, generate a file 'COMPONENT.jsx' with only the React JSX skeleton."
                    "Return only the code, no commentary."
                ),
            )
            style = Agent(
                name="style_agent",
                instructions=(
                    "Based on the Figma JSON and image, generate a file 'COMPONENT.css' with CSS classes matching the JSX."
                    "Return only the CSS code."
                ),
            )
            layout = Agent(
                name="layout_agent",
                instructions=(
                    "Combine 'COMPONENT.jsx' and 'COMPONENT.css' into a final 'COMPONENT.jsx' file,"
                    "including import './COMPONENT.css' and the JSX referencing those classes. Return only the code."
                ),
            )
            payload = json.dumps({"json": info["json"], "image_url": info["image_url"]})
            jsx = Runner.run_sync(struct, payload).final_output
            css = Runner.run_sync(style, payload).final_output
            prompt = f"JSX:\n{jsx}\nCSS:\n{css}\nImageURL: {info['image_url']}"
            final_code = Runner.run_sync(layout, prompt).final_output
            st.session_state.generation_results[key] = {"jsx": final_code, "css": css}

    # Display generated code
    for name, files in st.session_state.generation_results.items():
        comp_name = name.replace(" ", "")
        st.markdown(f"## {name}")
        st.markdown(f"**{comp_name}.jsx**")
        st.code(files["jsx"], language="javascript")
        st.markdown(f"**{comp_name}.css**")
        st.code(files["css"], language="css")

    # Feedback loop
    feedback = st.text_area("Feedback (will refine component code):")
    if st.button("Submit Feedback") and feedback:
        for info in component_infos:
            key = info["name"]
            prev = st.session_state.generation_results.get(key, {})
            fb_agent = Agent(
                name="feedback_agent",
                instructions=(
                    f"User feedback: {feedback}. Given the original Figma JSON and image_url,"
                    "and current 'COMPONENT.jsx' and 'COMPONENT.css',"
                    "generate updated 'COMPONENT.jsx' and 'COMPONENT.css' files."
                    "Return only the two code blocks, clearly labeled."
                ),
            )
            combined = json.dumps(
                {
                    "json": info["json"],
                    "image_url": info["image_url"],
                    "jsx": prev.get("jsx", ""),
                    "css": prev.get("css", ""),
                    "feedback": feedback,
                }
            )
            updated = Runner.run_sync(fb_agent, combined).final_output
            parts = updated.split("```")
            if len(parts) >= 4:
                st.session_state.generation_results[key]["jsx"] = parts[1].split(
                    "\n", 1
                )[1]
                st.session_state.generation_results[key]["css"] = parts[3].split(
                    "\n", 1
                )[1]


if __name__ == "__main__":
    main()
