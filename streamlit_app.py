#
#
# PDF to Audio Converter
# This code can be used to convert PDFs into audio podcasts, lectures, summaries, and more. It uses OpenAI's GPT models for text generation and text-to-speech conversion.
#
# Original Source: https://github.com/lamm-mit/PDF2Audio
#

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

def read_readme():
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, "r") as file:
            content = file.read()
            # Use regex to remove metadata enclosed in -- ... --
            content = re.sub(r'--.*?--', '', content, flags=re.DOTALL)
            return content
    else:
        return "README.md not found. Please check the repository for more information."

# Define multiple sets of instruction templates
INSTRUCTION_TEMPLATES = {
################# PODCAST ##################
    "podcast": {
        "intro": """Your task is to take the input text provided and turn it into an lively, engaging, informative podcast dialogue, in the style of NPR. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience.

Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

The podcast should have around 20000 words.
""",
    },
################# MATERIAL DISCOVERY SUMMARY ##################
    "SciAgents material discovery summary": {
        "intro": """Your task is to take the input text provided and turn it into a lively, engaging conversation between a professor and a student in a panel discussion that describes a new material. The professor acts like Richard Feynman, but you never mention the name.

The input text is the result of a design developed by SciAgents, an AI tool for scientific discovery that has come up with a detailed materials design.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that could be discussed in a podcast.

Define all terms used carefully for a broad audience of listeners.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """Brainstorm creative ways to discuss the main topics and key points you identified in the material design summary, especially paying attention to design features developed by SciAgents. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your description should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting. You never refer to the podcast, you just discuss the discovery and you focus on the new material design only.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. The presentation must focus on the novel aspects of the material design, behavior, and all related aspects.

Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience, but make it detailed, logical, and technical so that it has all necessary aspects for listeners to understand the material and its unexpected properties.

Remember, this describes a design developed by SciAgents, and this must be explicitly stated for the listeners.

Never use made-up names for the hosts and guests, but make it an engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

Make the dialogue as long and detailed as possible with great scientific depth, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.

The conversation should have around 20000 words.
"""
    },
################# LECTURE ##################
    "lecture": {
        "intro": """You are Professor Richard Feynman. Your task is to develop a script for a lecture. You never mention your name.

The material covered in the lecture is based on the provided text.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be covered in the lecture.

Define all terms used carefully for a broad audience of students.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for a high quality presentation.",
        "scratch_pad": """
Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, examples, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your lecture should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the lecture here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it fun and exciting.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a very long, engaging, informative script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the students.

Include clear definitions and terms, and examples.

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you, the professor. Stay on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest lecture you can, while still communicating the key information from the input text in an engaging way.

At the end of the lecture, naturally summarize the main insights and takeaways from the lecture. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas covered in this lecture one last time before class is over.

The lecture should have around 20000 words.
""",
    },
################# SUMMARY ##################
        "summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues.

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow.

Naturally summarize the main insights and takeaways from the summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

The summary should have around 1024 words.
""",
    },
################# SHORT SUMMARY ##################
        "short summary": {
        "intro": """Your task is to develop a summary of a paper. You never mention your name.

Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points, identify definitions, and interesting facts that need to be summarized.

Define all terms used carefully for a broad audience.
""",
        "text_instructions": "First, carefully read through the input text and identify the main topics, key points, and key facts. Think about how you could present this information in an accurate summary.",
        "scratch_pad": """Brainstorm creative ways to present the main topics and key points you identified in the input text. Consider using analogies, examples, or hypothetical scenarios to make the content more relatable and engaging for listeners.

Keep in mind that your summary should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms. Define all terms used clearly and spend effort to explain the background.

Write your brainstorming ideas and a rough outline for the summary here. Be sure to note the key insights and takeaways you want to reiterate at the end.

Make sure to make it engaging and exciting.
""",
        "prelude": """Now that you have brainstormed ideas and created a rough outline, it is time to write the actual summary. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.
""",
        "dialog": """Write a a script here, based on the key points and creative ideas you came up with during the brainstorming session. Keep it concise, and use a conversational tone and include any necessary context or explanations to make the content accessible to the the audience.

Start your script by stating that this is a summary, referencing the title or headings in the input text. If the input text has no title, come up with a succinct summary of what is covered to open.

Include clear definitions and terms, and examples, of all key issues.

Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.

There is only one speaker, you. Stay on topic and maintaining an engaging flow.

Naturally summarize the main insights and takeaways from the short summary. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner.

The summary should have around 256 words.
""",
    },
}

# Function to update instruction fields based on template selection
def update_instructions(template):
    return (
        INSTRUCTION_TEMPLATES[template]["intro"],
        INSTRUCTION_TEMPLATES[template]["text_instructions"],
        INSTRUCTION_TEMPLATES[template]["scratch_pad"],
        INSTRUCTION_TEMPLATES[template]["prelude"],
        INSTRUCTION_TEMPLATES[template]["dialog"]
           )



STANDARD_TEXT_MODELS = [
    "o1-preview-2024-09-12",
    "o1-preview",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
    "o1-mini-2024-09-12",
    "o1-mini",
    "chatgpt-4o-latest",
    "gpt-4-turbo",
    "openai/custom_model",
]

a1="""
STANDARD_AUDIO_MODELS = [
    "tts-1",
    "tts-1-hd",
]

STANDARD_VOICES = [
    "alloy",
    "echo",
    "fable",
    "onyx",
    "nova",
    "shimmer",
]
"""

class DialogueItem(BaseModel):
    text: str
    speaker: Literal["speaker-1", "speaker-2", "speaker-3", "speaker-4"]

class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

        
def conditional_llm(model, api_base=None, api_key=None):
    """
    Conditionally apply the @llm decorator based on the api_base parameter.
    If api_base is provided, it applies the @llm decorator with api_base.
    Otherwise, it applies the @llm decorator without api_base.
    """
    def decorator(func):
        if api_base:
            return llm(model=model, api_base=api_base)(func)
        else:
            return llm(model=model, api_key=api_key)(func)
    return decorator

def generate_transcript(
    file,
    openai_api_key: str = None,
    text_model: str = "gpt-4o-mini",  # Updated to use GPT-4 with LangChain
    audio_model: str = "tts-1",
    speaker_1_voice: str = "alloy",
    speaker_2_voice: str = "echo",
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

