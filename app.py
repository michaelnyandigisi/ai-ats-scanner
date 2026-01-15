import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. HELPER FUNCTIONS ---

def extract_text_from_pdf(file):
    """Extracts text from the uploaded PDF file."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def clean_text(text):
    """Cleans text by removing special characters and making it lowercase."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep only letters and numbers
    return text.lower()

# --- 2. STREAMLIT UI SETUP ---
st.set_page_config(page_title="AI ATS Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI Resume ATS Scanner")
st.markdown("### Optimize your resume for the job you want.")
st.markdown("Upload your resume and paste a Job Description (JD) to see your match score and missing keywords.")

# Layout: Two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 1. Upload Resume")
    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")

with col2:
    st.subheader("📝 2. Job Description")
    job_description = st.text_area("Paste the JD text here...", height=300)

# --- 3. THE ANALYSIS LOGIC ---

if st.button("🚀 Run Analysis", type="primary"):
    if uploaded_file is not None and job_description:
        with st.spinner("Analyzing text patterns..."):
            
            # A. Extract Text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # B. Get the Match Score (Cosine Similarity)
            # We put both texts into a list
            text_list = [resume_text, job_description]
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(text_list)
            match_percentage = cosine_similarity(count_matrix)[0][1] * 100
            match_percentage = round(match_percentage, 2)

            # C. Keyword Analysis (Find what's missing)
            # 1. Clean and split texts into sets of unique words
            resume_words = set(clean_text(resume_text).split())
            jd_words = set(clean_text(job_description).split())

            # 2. Define "Stop Words" (common words we don't care about)
            stop_words = {
                "the", "and", "is", "in", "to", "of", "a", "for", "with", "on", "at", "by", 
                "an", "be", "this", "that", "it", "as", "from", "or", "are", "was", "were", 
                "will", "has", "have", "had", "but", "not", "if", "we", "you", "can", "may"
            }

            # 3. Find missing words (Words in JD but NOT in Resume, and NOT stop words)
            missing_keywords = []
            for word in jd_words:
                if word not in resume_words and word not in stop_words:
                    missing_keywords.append(word)

            # --- 4. DISPLAY RESULTS ---
            
            st.divider()
            
            # Result Columns
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.subheader("📊 Match Score")
                st.metric(label="ATS Similarity", value=f"{match_percentage}%")
                
                if match_percentage >= 75:
                    st.success("✅ Strong Match! Your resume is well-optimized.")
                elif match_percentage >= 50:
                    st.warning("⚠️ Partial Match. You have the basics, but need more details.")
                else:
                    st.error("❌ Low Match. Tailor your resume significantly for this role.")

            with res_col2:
                st.subheader("🔍 Missing Keywords")
                st.write("These words appear in the JD but are **missing** from your resume:")
                
                if missing_keywords:
                    # Show the top 15 missing keywords as little tags
                    st.write(", ".join([f"`{word}`" for word in list(missing_keywords)[:15]]))