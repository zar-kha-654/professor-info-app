import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import re
import time
from io import BytesIO


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-120b"

MAX_PAGES = 35
MAX_TEXT_PER_PAGE = 12000
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.4

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0.0.0 Safari/537.36"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="University Professor Info Extractor",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 University Information Extractor")

st.write(
    "Upload your spreadsheet, provide an institutional website, "
    "choose how many professors you need, and the app will collect "
    "information found on that website."
)

st.info(
    "The app only uses information it can find on the provided "
    "institutional website. It does not guess or invent missing information."
)


# ============================================================
# GROQ API
# ============================================================

def get_groq_api_key():
    """
    Gets the Groq API key from Streamlit Secrets.
    """

    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


def ask_groq(prompt):
    """
    Send a prompt to Groq using the OpenAI-compatible API.
    """

    api_key = get_groq_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found in Streamlit Secrets."
        )

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict information extraction assistant. "
                    "Use ONLY information explicitly present in the provided "
                    "website content. Never guess, infer, fabricate, or use "
                    "outside knowledge. If information is missing, return "
                    "an empty string."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "max_tokens": 12000
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Groq API error {response.status_code}: "
            f"{response.text}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]


# ============================================================
# WEBSITE HELPERS
# ============================================================

def normalize_url(url):
    """
    Make sure URL has http/https.
    """

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return url


def get_domain(url):
    """
    Get domain without www.
    """

    domain = urlparse(url).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def is_same_domain(url, original_domain):
    """
    Only allow URLs belonging to the supplied institution.
    """

    try:
        domain = get_domain(url)

        return (
            domain == original_domain
            or domain.endswith("." + original_domain)
        )

    except Exception:
        return False


def clean_text(text):
    """
    Clean extracted website text.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_page(url):
    """
    Download and extract readable text and links from a webpage.
    """

    try:

        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            return "", []

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "text/html" not in content_type:
            return "", []

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "noscript",
            "svg",
            "footer",
            "nav"
        ]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)

        text = clean_text(text)

        # Limit page size
        text = text[:MAX_TEXT_PER_PAGE]

        links = []

        for a in soup.find_all("a", href=True):

            href = a.get("href")

            if not href:
                continue

            absolute_url = urljoin(
                url,
                href
            )

            # Remove fragments
            absolute_url = absolute_url.split("#")[0]

            if is_same_domain(
                absolute_url,
                get_domain(url)
            ):
                links.append(absolute_url)

        return text, list(set(links))

    except Exception:
        return "", []


# ============================================================
# LINK PRIORITY
# ============================================================

def link_priority(url):
    """
    Give likely useful pages a higher crawl priority.
    """

    url_lower = url.lower()

    keywords = {
        "faculty": 10,
        "professor": 10,
        "people": 9,
        "staff": 9,
        "department": 8,
        "academics": 7,
        "research": 7,
        "graduate": 7,
        "program": 7,
        "admission": 8,
        "admissions": 8,
        "application": 9,
        "apply": 9,
        "deadline": 9,
        "fees": 8,
        "tuition": 7,
        "contact": 6,
        "directory": 9
    }

    score = 0

    for keyword, value in keywords.items():

        if keyword in url_lower:
            score += value

    return score


# ============================================================
# WEBSITE CRAWLER
# ============================================================

def crawl_website(start_url, max_pages=MAX_PAGES):

    start_url = normalize_url(start_url)

    original_domain = get_domain(start_url)

    visited = set()

    # Priority queue
    urls_to_visit = [start_url]

    pages = []

    progress = st.progress(0)

    status = st.empty()

    while urls_to_visit and len(pages) < max_pages:

        # Sort so useful pages are processed first
        urls_to_visit = sorted(
            list(set(urls_to_visit)),
            key=link_priority,
            reverse=True
        )

        current_url = urls_to_visit.pop(0)

        if current_url in visited:
            continue

        if not is_same_domain(
            current_url,
            original_domain
        ):
            continue

        visited.add(current_url)

        status.write(
            f"🔎 Crawling page {len(pages) + 1}/{max_pages}: "
            f"{current_url}"
        )

        text, links = extract_page(current_url)

        if text:

            pages.append({
                "url": current_url,
                "text": text
            })

        # Add new links
        for link in links:

            if link not in visited:
                urls_to_visit.append(link)

        progress.progress(
            min(len(pages) / max_pages, 1.0)
        )

        time.sleep(REQUEST_DELAY)

    progress.empty()
    status.empty()

    return pages


# ============================================================
# COMBINE WEBSITE CONTENT
# ============================================================

def build_website_context(pages):

    context_parts = []

    for i, page in enumerate(pages):

        part = f"""
========================
PAGE {i + 1}
URL: {page["url"]}
========================

{page["text"]}
"""

        context_parts.append(part)

    return "\n".join(context_parts)


# ============================================================
# JSON EXTRACTION
# ============================================================

def extract_json_from_response(response):

    response = response.strip()

    # Remove markdown code block if model returned one
    response = re.sub(
        r"```json\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(
        r"```\s*",
        "",
        response
    )

    response = response.strip()

    # Find first JSON array
    start = response.find("[")

    end = response.rfind("]")

    if start != -1 and end != -1:

        json_text = response[start:end + 1]

        try:
            return json.loads(json_text)

        except json.JSONDecodeError:
            pass

    # Try JSON object containing professors
    start = response.find("{")

    end = response.rfind("}")

    if start != -1 and end != -1:

        json_text = response[start:end + 1]

        try:

            data = json.loads(json_text)

            if isinstance(data, dict):

                if "professors" in data:
                    return data["professors"]

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The AI returned an invalid JSON response."
    )


# ============================================================
# AI EXTRACTION
# ============================================================

def extract_professors(
    website_context,
    columns,
    number_of_professors
):

    columns_text = "\n".join(
        f"- {column}"
        for column in columns
    )

    prompt = f"""
You are extracting information from an institutional university website.

The user uploaded a spreadsheet with these exact columns:

{columns_text}

The user wants exactly {number_of_professors} professor/faculty records.

Your task:

1. Find professor/faculty members from the supplied website content.
2. Return up to {number_of_professors} different professors.
3. Fill the requested spreadsheet columns.
4. Use ONLY information explicitly present in the website content.
5. Do NOT use your own knowledge.
6. Do NOT search or invent information.
7. Do NOT guess email addresses.
8. Do NOT guess university names.
9. Do NOT guess departments.
10. Do NOT guess application fees.
11. Do NOT guess deadlines.
12. If a field cannot be found, return an empty string.
13. Use the professor's actual profile/page URL when available for a website/link column.
14. If a column does not apply to a professor, use the relevant information from the institutional website if explicitly available.
15. Do not create extra columns.
16. Return ONLY valid JSON.
17. The JSON must be a list of objects.
18. Each object must contain exactly the spreadsheet column names.
19. Preserve the exact column names supplied by the user.

IMPORTANT:
The information must come ONLY from the supplied website content.

Spreadsheet columns:

{columns_text}

Return format:

[
  {{
    "Column 1": "value",
    "Column 2": "value"
  }}
]

Website content:

{website_context}
"""

    response = ask_groq(prompt)

    records = extract_json_from_response(response)

    if not isinstance(records, list):
        raise ValueError(
            "AI response did not contain a list of records."
        )

    return records


# ============================================================
# CLEAN AI RECORDS
# ============================================================

def clean_records(records, columns, number_requested):

    cleaned = []

    for record in records:

        if not isinstance(record, dict):
            continue

        new_record = {}

        for column in columns:

            value = record.get(column, "")

            if value is None:
                value = ""

            value = str(value).strip()

            new_record[column] = value

        # Don't add completely empty records
        if any(
            value.strip()
            for value in new_record.values()
        ):
            cleaned.append(new_record)

    return cleaned[:number_requested]


# ============================================================
# EXCEL CREATION
# ============================================================

def create_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Professor Information"
        )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    number_of_professors = st.number_input(
        "Number of professors",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    max_pages = st.slider(
        "Maximum website pages to scan",
        min_value=5,
        max_value=50,
        value=35
    )

    st.caption(
        "Higher page counts may take longer."
    )


# ============================================================
# FILE UPLOAD
# ============================================================

st.subheader("1️⃣ Upload Spreadsheet")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

if uploaded_file:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(uploaded_file)

        else:

            df = pd.read_excel(uploaded_file)

        st.success(
            f"Spreadsheet loaded: {uploaded_file.name}"
        )

        st.write("### Detected columns")

        st.write(
            list(df.columns)
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Could not read the spreadsheet: {e}"
        )

        st.stop()


# ============================================================
# WEBSITE URL
# ============================================================

st.subheader("2️⃣ University Website")

website_url = st.text_input(
    "Enter the institutional website URL",
    placeholder="https://exampleuniversity.edu"
)


# ============================================================
# START BUTTON
# ============================================================

st.subheader("3️⃣ Generate Spreadsheet")

if st.button(
    "🚀 Find Information",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not uploaded_file:

        st.error(
            "Please upload a spreadsheet first."
        )

        st.stop()

    if not website_url.strip():

        st.error(
            "Please enter the institutional website URL."
        )

        st.stop()

    if len(df.columns) == 0:

        st.error(
            "Your spreadsheet does not contain any columns."
        )

        st.stop()

    # --------------------------------------------------------
    # API CHECK
    # --------------------------------------------------------

    if not get_groq_api_key():

        st.error(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

        st.info(
            "Add GROQ_API_KEY to your Streamlit app secrets."
        )

        st.stop()

    # --------------------------------------------------------
    # Crawl website
    # --------------------------------------------------------

    st.write("## 🔎 Searching website")

    try:

        pages = crawl_website(
            website_url,
            max_pages=max_pages
        )

    except Exception as e:

        st.error(
            f"Website crawling failed: {e}"
        )

        st.stop()

    if not pages:

        st.error(
            "No readable pages were found on this website."
        )

        st.stop()

    st.success(
        f"Found {len(pages)} readable pages."
    )

    # --------------------------------------------------------
    # Show crawled pages
    # --------------------------------------------------------

    with st.expander(
        "View pages that were scanned"
    ):

        for page in pages:

            st.write(
                f"🔗 {page['url']}"
            )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    website_context = build_website_context(
        pages
    )

    # --------------------------------------------------------
    # AI extraction
    # --------------------------------------------------------

    st.write(
        "## 🤖 Extracting information"
    )

    extraction_status = st.empty()

    extraction_status.write(
        f"Groq {MODEL} is extracting "
        f"{number_of_professors} professor records..."
    )

    try:

        records = extract_professors(
            website_context=website_context,
            columns=list(df.columns),
            number_of_professors=number_of_professors
        )

    except Exception as e:

        extraction_status.empty()

        st.error(
            f"Information extraction failed: {e}"
        )

        st.stop()

    extraction_status.empty()

    # --------------------------------------------------------
    # Clean records
    # --------------------------------------------------------

    records = clean_records(
        records,
        list(df.columns),
        number_of_professors
    )

    if not records:

        st.warning(
            "No professor information could be extracted "
            "from the supplied website."
        )

        st.stop()

    # --------------------------------------------------------
    # Create output dataframe
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        records,
        columns=list(df.columns)
    )

    # --------------------------------------------------------
    # Show result
    # --------------------------------------------------------

    st.success(
        f"Successfully prepared {len(result_df)} records."
    )

    st.write("## 📊 Prepared Spreadsheet")

    st.dataframe(
        result_df,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    excel_file = create_excel(
        result_df
    )

    st.download_button(
        label="⬇️ Download Completed Excel",
        data=excel_file,
        file_name="professor_information.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )
