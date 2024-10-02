
import streamlit as st

import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal

from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type

from functools import wraps

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGSMITH_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_PROJECT']="gen-podcast"

from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llmModel = ChatOpenAI(model="gpt-4o-mini", max_tokens=3000)

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2", "speaker-3", "speaker-4"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

def generate_transcript(
    file,
    openai_api_key: str = None,
    text_model: str = "gpt-4o-mini",  # Updated to use GPT-4 with LangChain
    intro_instructions: str = '',
    text_instructions: str = '',
    scratch_pad_instructions: str = '',
    prelude_dialog: str = '',
    podcast_dialog_instructions: str = '',
    edited_transcript: str = None,
    user_feedback: str = None,
    original_text: str = None,
    debug = False,
) -> tuple:
    
    reader=PdfReader(file)
    combined_text="\n\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # Setup LangChain's OpenAI model
    openai_llm = ChatOpenAI(model_name=text_model, openai_api_key=openai_api_key)

    podcast_dict=INSTRUCTION_TEMPLATES['podcast']
    intro_instructions=podcast_dict['intro']
    text_instructions = podcast_dict['text_instructions']
    scratch_pad_instructions = podcast_dict['scratch_pad']
    prelude_dialog = podcast_dict['prelude']
    podcast_dialog_instructions = podcast_dict['dialog']
    #edited_transcript = podcast_dict['intro']
    #user_feedback = podcast_dict['intro']
    #original_text = podcast_dict['intro']

    # Construct the full prompt using LangChain's ChatPromptTemplate
    prompts =  [
        SystemMessage(content=intro_instructions),
        HumanMessage(f"Here is the original input text:\n{combined_text}"),
        HumanMessage(text_instructions),
        HumanMessage(f"Brainstorm: {scratch_pad_instructions}"),
        HumanMessage(f"Prelude: {prelude_dialog}"),
        HumanMessage(f"Dialogue: {podcast_dialog_instructions}"),
        HumanMessage(f"Edits: {edited_transcript if edited_transcript else ''}"),
        HumanMessage(f"User feedback: {user_feedback if user_feedback else ''}")
    ]

    # Generate the dialogue using LangChain's OpenAI call
    llm_output = openai_llm.with_structured_output(Dialogue).invoke(prompts)

    # Process the response
    generated_dialogue = llm_output.dialogue

    print(f"Generated dialogue: {generated_dialogue}")

    # Generate audio from the transcript
    transcript = ""
    characters = 0
    for gd in generated_dialogue:
        one_line=f"{gd.speaker}: {gd.text}"
        characters += len(one_line)
        transcript +=one_line+"\n\n"

    logger.info(f"Generated {characters} characters of audio")
    return transcript, combined_text

st.markdown("# Upload file: PDF")
uploaded_file=st.file_uploader("Upload PDF file",type="pdf")

if uploaded_file is not None and st.button("Generate"):
    os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']
    #transcript, original_text=generate_transcript(['Testfile.pdf'])
    transcript, original_text=generate_transcript(uploaded_file)
    #print(transcript)
    st.write(transcript)