# AI Recruiter Assistant

An intelligent Streamlit application that helps recruiters analyze job descriptions, generate recruitment materials, and evaluate candidates using Google's Gemini AI.

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
3. Get a Gemini API key from Google AI Studio

4. Run the Streamlit application:

```bash
streamlit run app.py
```
## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
