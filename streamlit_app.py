from typing import List
import streamlit as st
import tempfile
import json
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate
from llama_index import (
    GPTSimpleVectorIndex,
    BeautifulSoupWebReader,
    LLMPredictor,
    download_loader,
)

# Data processing
llm, embeddings, llm_predictor = None, None, None


@st.cache_resource
def init_llm(openai_api_key):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    llm = OpenAI(
        temperature=0.7,
        max_tokens=1000,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    llm_predictor = LLMPredictor(llm)
    return llm, embeddings, llm_predictor


def build_qa_chain(docs):
    return GPTSimpleVectorIndex(docs, llm_predictor=llm_predictor)


PDFReader = download_loader("PDFReader")
StringIterableReader = download_loader("StringIterableReader")


def load_pdf(pdf):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf.read())
        pdf = f.name
    loader = PDFReader()
    return loader.load_data(file=pdf)


def load_url(url):
    loader = BeautifulSoupWebReader()
    return loader.load_data(urls=[url])


def load_str(strs: List[str]):
    loader = StringIterableReader()
    return loader.load_data(texts=strs)


@st.cache_data
def run_user_profile_chain(resume_pdf, resume_url):
    resume_docs = []
    if resume_pdf:
        resume_docs.extend(load_pdf(resume_pdf))
    if resume_url:
        resume_docs.extend(load_url(resume_url))
    # st.write("resume_docs", resume_docs)
    resume_chain = build_qa_chain(resume_docs)
    answer = resume_chain.query(
        """What are the user's name, skills, education, experiences and professional highlights? Format the result as a JSON object with 5 keys: "name", "skills", "education", "experiences" and "highlights". Values are all strings.
        
        Result:
        """,
        similarity_top_k=1,
    )
    try:
        result = json.loads(answer.response)
        st.write("result", result)
        return result
    except json.decoder.JSONDecodeError as e:
        st.write("json parsing failed!", answer)
        raise e


@st.cache_data
def run_company_chain(job_description_url, company_info_url, job_info_text):
    company_docs = []
    if job_description_url:
        company_docs.extend(load_url(job_description_url))
    if company_info_url:
        company_docs.extend(load_url(company_info_url))
    if job_info_text:
        company_docs.extend(load_str([job_info_text]))

    # st.write("company_docs", company_docs)
    company_chain = build_qa_chain(company_docs)
    answer = company_chain.query(
        """
You are a job seeker researching a company. You want to infer and extract the company's mission, values and culture.

Expected format is a JSON object with 4 keys: "company_name" (string), "position_name" (string), "job_description" (string that includes a short summary of qualifications) and "values" (an array of strings). For each item in the "values" array, the value should be a string that includes the description of the company value and your reasoning and a quote from the company's website. Add company mission and culture into "values" too.

Example: {"company_name": "Google", "position_name": "Software Engineer", "job_description": "We are looking for a software engineer to join our team.", "values": ["We value diversity", "We value work-life balance"]}

Result:
{        
""",
        similarity_top_k=1,
    )
    try:
        result = json.loads("{" + answer.response)
        st.write("result", result)
        return result
    except json.decoder.JSONDecodeError as e:
        st.write("json parsing failed!", answer)
        raise e


def generate_cover_letter(user_profile, company_values, job_description):
    st.write("Generating cover letter...")
    prompt = PromptTemplate(
        input_variables=["job_description", "resume", "company_values", "user_name"],
        template="""
Your name is {user_name} and you are applying for the position as described in the context. Write a cover letter that reflects the company's values in the context below. Include the user's real name and professional highlights.

- Instead of listing all skills and experiences, summarize them into a few sentences.
- DO NOT copy the qualifications in the job description. 
- DO NOT include previous employer names.

###

Job Description: {job_description}

Company values: {company_values}

Resume: {resume}

###

Cover letter:


""",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    cover_letter = llm_chain.predict(
        verbose=True,
        job_description=job_description,
        resume=user_profile,
        user_name=user_profile["name"],
        company_values="\n".join(company_values),
    )
    st.write("Done!")
    st.session_state["cover-letter"] = cover_letter


def get_feedback(cover_letter, job_description):
    st.write("Generating feedback...")
    prompt = PromptTemplate(
        input_variables=["cover_letter", "job_description"],
        template="""
You are a hiring manager at the company. What feedback do you have for the cover letter below? Focuse on providing critical and constructive feedback. Be as specific as possible. Include quotes from the cover letter when necessary.
        
# Context

## Job Description
{job_description}

## Cover letter
{cover_letter}

# Feedback:
""",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    feedback = llm_chain.predict(
        verbose=True,
        cover_letter=cover_letter,
        job_description=job_description,
    )
    return feedback


# Streamlit UI

"""
# CoverLetter.ai
Research your dream job and write your cover letter in minutes.
"""

openai_api_key = st.text_input("Enter your OpenAI API key")
if openai_api_key:
    llm, embeddings, llm_predictor = init_llm(openai_api_key)

"""
## About you
"""
resume_url = st.text_input("Enter your resume URL")
resume_pdf = st.file_uploader("Or, upload your resume PDF", type=["pdf"])
user_profile, company_profile = None, None
if resume_pdf or resume_url:
    user_profile = run_user_profile_chain(resume_pdf, resume_url)

"""
## Your dream job
"""
job_description_url, company_info_url, job_info_text = None, None, None
job_description_url = st.text_input("Enter the job description URL")
company_info_url = st.text_input(
    "Enter the company info URL (typically the About page)"
)
job_info_text = st.text_area(
    "Paste anything about the job or the company here", height=200
)
if job_info_text or job_description_url or company_info_url:
    company_profile = run_company_chain(
        job_description_url, company_info_url, job_info_text
    )

"""
## Customize it!
Select the company values you want to include in your cover letter.
"""

user_items = []
company_items = []
if company_profile and "values" in company_profile:
    for value in company_profile["values"]:
        selected = st.checkbox(value)
        if selected:
            company_items.append(value)

# notes = st.text_area("Notes", placeholder="Enter any notes you want to include in the cover letter", height=50)
generate_clicked = st.button("Generate Cover Letter")

if generate_clicked:
    generate_cover_letter(
        user_profile=user_profile,
        company_values=company_items,
        job_description=company_profile["job_description"],
    )

st.header("Your cover letter")
st.text_area(
    "", placeholder="Your cover letter will appear here", key="cover-letter", height=500
)

"""
## Critiques
"""
feedback_clicked = st.button("Get feedback")
if feedback_clicked:
    feedback = get_feedback(
        st.session_state["cover-letter"],
        job_description=company_profile["job_description"],
    )
    st.write(feedback)
