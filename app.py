import streamlit as st
import os
import tempfile
import google.generativeai as genai
from PyPDF2 import PdfReader
import docx
import io
import re
from pathlib import Path

# Configure the Gemini API
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Function to read text from a PDF file
def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to read text from a DOCX file
def read_docx(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    # Get the file extension
    file_extension = Path(uploaded_file.name).suffix.lower()
    
    # Read the file into a bytes buffer
    bytes_data = uploaded_file.getvalue()
    
    # Create a BytesIO object
    file_io = io.BytesIO(bytes_data)
    
    # Extract text based on file type
    if file_extension == ".pdf":
        return read_pdf(file_io)
    elif file_extension in [".docx", ".doc"]:
        return read_docx(file_io)
    elif file_extension == ".txt":
        return uploaded_file.getvalue().decode("utf-8")
    else:
        return "Unsupported file format. Please upload a PDF, DOCX, or TXT file."

# Function to analyze job description
def analyze_job_description(jd_text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Analyze the following job description and extract the information in a structured format:
    
    {jd_text}
    
    Provide the following information:
    1. Role Summary: A concise summary of the role.
    2. Must-Have Skills: List the essential skills and qualifications.
    3. Good-to-Have Skills: List the preferred but not mandatory skills.
    4. Experience Required: The required years of experience and specific experience areas.
    5. Target Company Types: Types of companies where potential candidates might be found (e.g., SaaS, startup, product-based, service-based, MNC).
    6. LinkedIn Xray Search: Create 5 search strings for Google to find candidates on LinkedIn.
    7. Interview Questions: 10 specific questions to ask during candidate screening.
    8. Outreach Email Template: A template for reaching out to potential candidates.
    
    Format the response in Markdown for better readability.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Function for the chat feature
def chat_with_jd_context(jd_text, user_query):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are an AI assistant for recruiters. You have the following job description:
    
    {jd_text}
    
    The recruiter asks: {user_query}
    
    Provide a helpful, well-structured, and context-aware response. If they are asking about recruitment strategies, candidate sourcing, or interview techniques specific to this role, provide detailed guidance. If they're asking about the job description itself, analyze it and provide insights.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Function to evaluate resume against job description
def evaluate_resume(jd_text, resume_text):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    Evaluate the following resume against the job description. Provide a detailed analysis including:
    
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text}
    
    Please provide:
    
    1. Overall Suitability Score (0-100)
    2. Skills Match Analysis:
       - Must-have skills present
       - Must-have skills missing
       - Good-to-have skills present
    3. Experience Analysis:
       - Relevant experience
       - Company type match (e.g., SaaS, startup experience if required)
       - Project complexity assessment
    4. Red Flags:
       - Employment gaps
       - Job hopping
       - Mismatched career progression
    5. Key Strengths for this role
    6. Key Weaknesses for this role
    7. Recommendation: Should proceed to interview? (Yes/No/Maybe with explanation)
    
    Format the response in Markdown for better readability.
    """
    
    response = model.generate_content(prompt)
    return response.text

# Main application
def main():
    st.set_page_config(page_title="AI Recruiter Assistant", page_icon="👔", layout="wide")
    
    st.title("AI Recruiter Assistant")
    st.markdown("Upload a job description and get AI-powered recruitment assistance.")
    
    # Sidebar for API key
    with st.sidebar:
        st.title("Configuration")
        api_key = st.text_input("Enter Gemini API Key", type="password")
        if api_key:
            configure_genai(api_key)
            st.success("API key configured!")
        else:
            st.warning("Please enter your Gemini API key to use the application.")
    
    # Check if API key is provided
    if not api_key:
        st.info("Please enter your Gemini API Key in the sidebar to get started.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["JD Analysis", "Chat with JD", "Resume Evaluation"])
    
    # Job Description Analysis Tab
    with tab1:
        st.header("Job Description Analysis")
        
        uploaded_jd = st.file_uploader("Upload Job Description", type=["pdf", "docx", "doc", "txt"], key="jd_upload")
        
        jd_text = ""
        if uploaded_jd is not None:
            jd_text = extract_text_from_file(uploaded_jd)
            with st.expander("View Extracted Text"):
                st.text(jd_text)
            
            if st.button("Analyze Job Description"):
                with st.spinner("Analyzing job description..."):
                    analysis = analyze_job_description(jd_text)
                    st.markdown(analysis)
    
    # Chat with JD Context Tab
    with tab2:
        st.header("Chat with JD Context")
        
        uploaded_jd_chat = st.file_uploader("Upload Job Description", type=["pdf", "docx", "doc", "txt"], key="jd_chat_upload")
        
        jd_text_chat = ""
        if uploaded_jd_chat is not None:
            jd_text_chat = extract_text_from_file(uploaded_jd_chat)
            with st.expander("View Extracted Text"):
                st.text(jd_text_chat)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Accept user input
            if prompt := st.chat_input("Ask about the job description..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_with_jd_context(jd_text_chat, prompt)
                        st.markdown(response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Resume Evaluation Tab
    with tab3:
        st.header("Resume Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_jd_eval = st.file_uploader("Upload Job Description", type=["pdf", "docx", "doc", "txt"], key="jd_eval_upload")
            jd_text_eval = ""
            if uploaded_jd_eval is not None:
                jd_text_eval = extract_text_from_file(uploaded_jd_eval)
                with st.expander("View Extracted JD Text"):
                    st.text(jd_text_eval)
        
        with col2:
            uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "doc", "txt"])
            resume_text = ""
            if uploaded_resume is not None:
                resume_text = extract_text_from_file(uploaded_resume)
                with st.expander("View Extracted Resume Text"):
                    st.text(resume_text)
        
        if jd_text_eval and resume_text:
            if st.button("Evaluate Resume"):
                with st.spinner("Evaluating resume against job description..."):
                    evaluation = evaluate_resume(jd_text_eval, resume_text)
                    st.markdown(evaluation)

if __name__ == "__main__":
    main()