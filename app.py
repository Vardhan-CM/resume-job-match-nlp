import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load text files
with open("sample_resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

with open("sample_jobdesc.txt", "r", encoding="utf-8") as f:
    jobdesc_text = f.read()

st.title("🧠 Resume vs Job Description Keyword Matcher")

st.subheader("📄 Resume Text")
st.text_area("Resume", resume_text, height=200)

st.subheader("📋 Job Description Text")
st.text_area("Job Description", jobdesc_text, height=200)

# TF-IDF Matching
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform([resume_text, jobdesc_text])
similarity_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

st.metric(label="🔍 Match Score", value=f"{similarity_score * 100:.2f}%")

# Highlight missing keywords
resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
job_words = set(re.findall(r'\b\w+\b', jobdesc_text.lower()))
missing_keywords = job_words - resume_words

st.subheader("❗ Missing Keywords from Resume")
if missing_keywords:
    st.write(", ".join(sorted(missing_keywords)))
else:
    st.write("✅ All key terms covered!")
