import streamlit as st
import os

def main():
    stop_langsmith_trace="""
    os.environ["LANGCHAIN_TRACING_V2"]="true"
    os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
    os.environ["LANGSMITH_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
    os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
    os.environ['LANGCHAIN_PROJECT']="yt-review"
    """
    
    home_pages=[
        st.Page("streamlit_app.py", title="Home", icon="🏠"),
        #st.Page("ui/view_uploads.py", title="View Uploads", icon="1️⃣"),
        #st.Page("ui/41_one_video_e2e.py", title="Process One Video", icon="2️⃣"),
        #st.Page("ui/14_transcript_yt_video.py", title="Transcript YT Video", icon="2️⃣"),
        #st.Page("ui/11_transcripts_with_answers.py", title="Transcript Q&A", icon="🌎"),
    ]


    podcast_pages=[
        st.Page("ui/original.py", title="Original version", icon="1️⃣"),
        st.Page("ui/notebookllm_prompt.py", title="Using NotebooLLM prompt", icon="1️⃣"),
        #st.Page("ui/process_pdf_text.py", title="PDF text", icon="2️⃣"),
        #st.Page("ui/process_pdf_image.py", title="PDF image", icon="2️⃣"),
        #st.Page("ui/extract_pdf_text.py", title="PDF-to-Text", icon="2️⃣"),
    ]

    skip_groups="""

    ref_pages=[
        st.Page("ui/process_ref.py", title="Ref", icon="2️⃣"),
    ]

    stat_pages=[
        st.Page("ui/stats.py", title="Interesting data points", icon="🌎"),
        st.Page("ui/test.py", title="Testing 1-2-3", icon="1️⃣"),
        #st.Page("ui/2_transcribe_videos.py", title="Transcribe", icon="2️⃣"),
        #st.Page("ui/3_qna.py", title="Q & A", icon="🌎"),
    ]

    classify_pages=[
        st.Page("ui/process_classify.py", title="Classify 1", icon="1️⃣"),
        st.Page("ui/process_classify2.py", title="Classify 2", icon="2️⃣"),
        st.Page("ui/process_classify3.py", title="Classify 3", icon="🌎"),
        st.Page("ui/process_data4.py", title="Data 4", icon="🌎"),
        st.Page("ui/process_classify5.py", title="Classify 5", icon="🌎"),
        st.Page("ui/process_data6.py", title="Data 6", icon="🌎"),
    ]

    extract_pages=[
        st.Page("ui/process_extract1.py", title="Extract 1", icon="1️⃣"),
        st.Page("ui/process_extract2.py", title="Extract 2", icon="1️⃣"),
    ]
    """

    pages={
        "Home": home_pages,
        "Podcast": podcast_pages,
        #"Process reference": ref_pages,
        #"Statistics": stat_pages,
        #"Classify": classify_pages,
        #"Extract": extract_pages,
    }
    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

