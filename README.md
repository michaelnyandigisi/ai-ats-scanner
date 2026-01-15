# AI Resume ATS Scanner

A Full-Stack Python application that uses Machine Learning techniques to compare a Resume (PDF) against a Job Description. It provides a similarity score and identifies missing keywords to help job seekers bypass Applicant Tracking Systems (ATS).



## Features
- **PDF Text Extraction:** Automatically parses content from uploaded PDF resumes.
- **AI-Powered Analysis:** Uses `Natural Language Processing (NLP)` and `Cosine Similarity` to calculate the match percentage.
- **Keyword Gap Analysis:** Identifies specific industry terms and keywords missing from the resume.
- **Interactive UI:** Built with Streamlit for a clean, user-friendly experience.

## 🛠️ Tech Stack
- **Language:** Python 3.9+
- **Frontend:** Streamlit
- **ML/NLP:** Scikit-Learn (CountVectorizer)
- **PDF Library:** PyPDF2

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/michaelnyandigisi/ai-ats-scanner.git](https://github.com/michaelnyandigisi/ai-ats-scanner.git)
   cd ai-ats-scanner

2. **Create a Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Run the App:**
    ```bash
    pip install -r requirements.txt


**License**
Distributed under the MIT License.
A vibe code by MichaelNyandigisi


