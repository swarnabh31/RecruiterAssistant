# AI Recruiter Assistant

An intelligent Streamlit application that helps recruiters analyze job descriptions, generate recruitment materials, and evaluate candidates using local LLMs via Ollama.

## Features

- **Job Description Analysis**: Upload a JD and automatically extract:
  - Role Summary
  - Must-Have Skills
  - Good-to-Have Skills
  - Experience Required
  - Target Company Types
  - LinkedIn X-Ray Search Strings
  - Interview Questions
  - Outreach Email Templates

- **Interactive Chat**: Ask questions about the job description, recruitment strategies, or get clarifications with context-aware responses.

- **Resume Evaluation**: Compare candidate resumes against job descriptions with detailed analysis:
  - Overall Suitability Score
  - Skills Match Analysis
  - Experience Analysis
  - Red Flags Detection
  - Strengths and Weaknesses
  - Hiring Recommendations

## Prerequisites

- [Ollama](https://ollama.ai) installed and running
- At least one model pulled (e.g., `ollama pull llama3.2`)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/swarnabh31/RecruiterAssistant.git
cd RecruiterAssistant
```
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running and has models available:
```bash
ollama serve
ollama pull llama3.2
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. Launch the app with `streamlit run app.py`
2. The app automatically detects all locally available Ollama models
3. Select your preferred model from the dropdown in the sidebar
4. Upload a job description and use the tools

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
