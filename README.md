# 🎓 University Professor Information Extractor

A simple Streamlit application that extracts professor and university information from an institutional website and fills an uploaded spreadsheet.

## Features

- Upload Excel or CSV spreadsheet
- Automatically detects spreadsheet column names
- Provide an institutional university website
- Choose how many professor records you need
- Crawls pages within the provided institution's domain
- Uses Groq `openai/gpt-oss-120b`
- Extracts only information found on the website
- Does not intentionally guess or invent missing information
- Leaves unavailable fields blank
- Download completed spreadsheet as Excel

## Tech Stack

- Python
- Streamlit
- Groq API
- BeautifulSoup
- Requests
- Pandas
- OpenPyXL

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
