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
        """What are the user's name, skills, education, experiences and professional highlights? Format the result as a JSON object with 5 keys: "name", "skills", "education", "experiences" and "highlights". Values are all strings.""",
        similarity_top_k=1,
    )
    try:
        result = json.loads(answer.response)
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
Extract the job and company information as a JSON object with 4 keys: "company_name" (string), "position_name" (string), "job_description" (string) and "values" (an array of strings). For each item in the "values" array, the value should be a string that includes the description of the company value and your reasoning and a quote from the company's website. 

Example: {"company_name": "Google", "position_name": "Software Engineer", "job_description": "We are looking for a software engineer to join our team.", "values": ["We value diversity", "We value work-life balance"]}

Result:
        
""",
        similarity_top_k=1,
    )
    try:
        result = json.loads(answer.response)
        # st.write("result", result)
        return result
    except json.decoder.JSONDecodeError as e:
        st.write("json parsing failed!", answer)
        raise e


def generate_cover_letter(user_profile, company_values, job_description):
    st.write("Generating cover letter...")
    prompt = PromptTemplate(
        input_variables=["job_description", "resume", "company_values"],
        template="""
Write a cover letter that reflects the company's values in the context below. DO NOT copy information from word to word. Rephrase the information in your own words. Include the user's real name and professional highlights.

- Do not list all skills and experiences. 
- DO NOT copy the qualifications in the job description. 
- DO NOT include previous employer names.

# Context

## Job Description
{job_description}

## Company values
{company_values}

## Resume
{resume}

# Cover letter:


""",
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    cover_letter = llm_chain.run(
        verbose=True,
        job_description=job_description,
        resume=user_profile,
        company_values="\n".join(company_values),
    )
    st.write("Done!")
    st.session_state["cover-letter"] = cover_letter


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
# openai_api_key = st.text_input("Enter your OpenAI API key")
resume_url = st.text_input("Enter your resume URL")
resume_pdf = st.file_uploader("Or, upload your resume PDF", type=["pdf"])
user_profile, company_profile = None, None
if resume_pdf or resume_url:
    user_profile = run_user_profile_chain(resume_pdf, resume_url)

"""
## Your dream job
"""
url_or_text = st.radio("Pick one", ["Text", "URLs"])
job_description_url, company_info_url, job_info_text = None, None, None
if url_or_text == "Text":
    job_info_text = st.text_area("Paste the job description here", height=200)
if url_or_text == "URLs":
    job_description_url = st.text_input("Enter the job description URL")
    company_info_url = st.text_input(
        "Enter the company info URL (typically the About page)"
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
