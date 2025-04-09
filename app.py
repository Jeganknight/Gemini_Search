import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
from pydantic import BaseModel

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Queries(BaseModel):
    wide_queries: list[str]
    deep_queries: list[str]

def generate_queries(topic, width, depth, deepdive_topic):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"You are an expert in topic:{topic}. Your task is to generate search queries to aid the user's research on said topic. The user will provide you research width and depth. Width indicates how wide the research needs to be. Depth indicates how deep the research needs to be for a specific topic. You need to generate {width} search queries to cover the width of the research and {depth} search queries to go deeper into the subtopic: {deepdive_topic}.",
        config={
            'response_mime_type': 'application/json',
            'response_schema': list[Queries],
        }
    )
    return json.loads(response.text)

def generate_context(search_query: str):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"{search_query}",
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                google_search=types.GoogleSearchRetrieval
            )]
        )
    )
    return response.text, response.candidates[0].grounding_metadata.grounding_chunks

def generate_report(topic, subtopic):
    """Main function for generating a report based on Topic and Subtopic only."""

    # Fixed width & depth internally
    width = 3
    depth = 3

    sys_instruct = f"""
You are an expert analyst in the topic: {topic}.
Your task is to answer the specific subtopic/question provided: {subtopic}.
Provide only the direct answer that satisfies the user's request — in a clear, concise, and structured manner — without adding any unnecessary background or unrelated information.
If the subtopic requires data (like lists, tables, rankings, stats, etc.), present them clearly in a well-formatted table or list.
At the end of your response, include a 'References' section with proper hyperlinks from the Citations object provided. Each citation has a title and a URI — ensure each is hyperlinked correctly so the user can verify the source.
Strictly stick to answering only what is asked in {subtopic}.
"""

    search_queries = generate_queries(topic, subtopic, width, depth)

    total_context = []
    sources = []

    for query in search_queries:
        context, source_list = generate_context(query)
        total_context.append(context)
        for src in source_list:
            sources.append({"title": src.web.title, "uri": src.web.uri})

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=sys_instruct
        ),
        contents=f"Context: {json.dumps(total_context)} Citations: {json.dumps(sources)}"
    )

    return response.text

# Streamlit app
st.title(" Gemini Beacon ")

topic = st.text_input("Enter the Main Topic (e.g., IPL 2025 / Stock Price):")
subtopic = st.text_input("Enter the Exact Question/Subtopic (e.g., Current Top Run Scorers / Current Stock Price Of Amazon):")

if st.button("Generate Report"):
    if topic and subtopic:
        with st.spinner("Generating report..."):
            final_report = generate_report(topic, subtopic)
        st.markdown(final_report, unsafe_allow_html=True)
    else:
        st.error("Please provide both a topic and a subtopic.")
