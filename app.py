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

REQUEST_TIMEOUT = 20
MAX_PAGES = 40
MAX_TEXT_PER_PAGE = 15000
REQUEST_DELAY = 0.3

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/153.0.0.0 Safari/537.36"
)


# ============================================================
# STREAMLIT
# ============================================================

st.set_page_config(
    page_title="University Information Extractor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 University Information Extractor")

st.write(
    "Upload a spreadsheet, provide an institutional website, "
    "choose how many records you need, and generate a completed spreadsheet."
)

st.info(
    "The AI is instructed to use only information explicitly found "
    "on the provided website. Missing information is left blank."
)


# ============================================================
# GROQ
# ============================================================

def get_api_key():

    try:
        return st.secrets["GROQ_API_KEY"]

    except Exception:
        return None


def ask_groq(prompt):

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
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
                    "You are a strict website information extraction "
                    "assistant. Use ONLY the information provided in "
                    "the website content. Never guess, infer, fabricate, "
                    "or use outside knowledge. If information is not "
                    "explicitly available, return an empty string."
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
# URL HELPERS
# ============================================================

def normalize_url(url):

    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return url.rstrip("/")


def get_domain(url):

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def same_domain(url, original_domain):

    try:

        domain = get_domain(url)

        return (
            domain == original_domain
            or domain.endswith("." + original_domain)
        )

    except Exception:

        return False


# ============================================================
# PAGE DOWNLOAD
# ============================================================

def download_page(url):

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,"
                    "application/xml;q=0.9,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9"
            },
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )

        if response.status_code != 200:

            return "", []

        # IMPORTANT:
        # Do not reject the page just because Content-Type
        # is unusual or missing.

        content = response.text

        if not content:

            return "", []

        soup = BeautifulSoup(
            content,
            "html.parser"
        )

        # Remove things that aren't useful for extraction
        for tag in soup.find_all([
            "script",
            "style",
            "noscript",
            "svg"
        ]):

            tag.decompose()

        # Extract text
        text = soup.get_text(
            " ",
            strip=True
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        text = text[:MAX_TEXT_PER_PAGE]

        # Extract links
        links = []

        for tag in soup.find_all("a", href=True):

            href = tag.get("href")

            if not href:
                continue

            # Ignore non-web links
            if href.startswith((
                "mailto:",
                "tel:",
                "javascript:",
                "#"
            )):
                continue

            absolute = urljoin(
                response.url,
                href
            )

            # Remove fragments
            absolute = absolute.split("#")[0]

            links.append(absolute)

        return text, list(set(links))

    except Exception as e:

        return "", []


# ============================================================
# LINK PRIORITY
# ============================================================

def score_url(url):

    url_lower = url.lower()

    keywords = [
        ("faculty", 20),
        ("people", 20),
        ("professor", 20),
        ("academic", 15),
        ("academics", 15),
        ("staff", 15),
        ("directory", 15),
        ("department", 12),
        ("school", 10),
        ("research", 10),
        ("program", 10),
        ("programs", 10),
        ("admission", 12),
        ("admissions", 12),
        ("application", 12),
        ("apply", 12),
        ("deadline", 10),
        ("fee", 10),
        ("fees", 10),
        ("tuition", 8),
        ("contact", 5)
    ]

    score = 0

    for keyword, points in keywords:

        if keyword in url_lower:
            score += points

    return score


# ============================================================
# WEBSITE CRAWLER
# ============================================================

def crawl_website(start_url, max_pages):

    start_url = normalize_url(start_url)

    original_domain = get_domain(start_url)

    visited = set()

    queue = [
        start_url
    ]

    pages = []

    progress = st.progress(0)

    status = st.empty()

    while queue and len(pages) < max_pages:

        # Highest priority first
        queue = sorted(
            list(set(queue)),
            key=score_url,
            reverse=True
        )

        current_url = queue.pop(0)

        if current_url in visited:
            continue

        if not same_domain(
            current_url,
            original_domain
        ):
            continue

        visited.add(current_url)

        status.write(
            f"🔎 Scanning {len(pages) + 1}/{max_pages}: "
            f"{current_url}"
        )

        text, links = download_page(
            current_url
        )

        if text:

            pages.append({
                "url": current_url,
                "text": text
            })

        # Add useful same-domain links
        for link in links:

            if link in visited:
                continue

            if not same_domain(
                link,
                original_domain
            ):
                continue

            # Avoid obvious files
            lower = link.lower()

            if lower.endswith((
                ".pdf",
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".zip",
                ".doc",
                ".docx",
                ".xls",
                ".xlsx"
            )):
                continue

            queue.append(link)

        progress.progress(
            min(
                len(pages) / max_pages,
                1.0
            )
        )

        time.sleep(
            REQUEST_DELAY
        )

    progress.empty()
    status.empty()

    return pages


# ============================================================
# WEBSITE CONTEXT
# ============================================================

def build_context(pages):

    pieces = []

    for number, page in enumerate(pages, 1):

        pieces.append(
            f"""
==================================================
PAGE {number}
URL: {page["url"]}
==================================================

{page["text"]}
"""
        )

    return "\n".join(pieces)


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(response):

    response = response.strip()

    # Remove markdown fences
    response = re.sub(
        r"```json",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = response.replace(
        "```",
        ""
    ).strip()

    # Try array
    start = response.find("[")
    end = response.rfind("]")

    if start != -1 and end != -1:

        candidate = response[
            start:end + 1
        ]

        try:
            return json.loads(candidate)

        except Exception:
            pass

    # Try object
    start = response.find("{")
    end = response.rfind("}")

    if start != -1 and end != -1:

        candidate = response[
            start:end + 1
        ]

        try:

            obj = json.loads(candidate)

            if "professors" in obj:
                return obj["professors"]

        except Exception:
            pass

    raise ValueError(
        "The AI did not return valid JSON."
    )


# ============================================================
# AI EXTRACTION
# ============================================================

def extract_information(
    website_context,
    columns,
    number_needed
):

    column_text = "\n".join(
        f"- {column}"
        for column in columns
    )

    prompt = f"""
Extract information from the institutional website content below.

The spreadsheet contains these EXACT columns:

{column_text}

The user wants up to {number_needed} professor/faculty records.

RULES:

1. Use ONLY information explicitly present in the website content.
2. Never use your general knowledge.
3. Never guess.
4. Never infer an email address.
5. Never invent an application fee.
6. Never invent an application deadline.
7. Never invent a department.
8. Never invent a university name.
9. If a field is unavailable, use "".
10. Return at most {number_needed} people.
11. Prefer actual academic/professor/faculty members.
12. Do not include students unless the website explicitly identifies them
    as faculty/professors.
13. Do not include advisory board members unless they are clearly
    university academic staff.
14. Preserve the exact spreadsheet column names.
15. Do not create additional columns.
16. Return ONLY valid JSON.

For website/profile links, use a URL explicitly present in the supplied
website content.

Return exactly this structure:

[
  {{
    "column name": "value"
  }}
]

Spreadsheet columns:

{column_text}

WEBSITE CONTENT:

{website_context}
"""

    response = ask_groq(
        prompt
    )

    return parse_json(
        response
    )


# ============================================================
# CLEAN RESULTS
# ============================================================

def clean_results(
    records,
    columns,
    number_needed
):

    final = []

    for record in records:

        if not isinstance(
            record,
            dict
        ):
            continue

        row = {}

        for column in columns:

            value = record.get(
                column,
                ""
            )

            if value is None:
                value = ""

            row[column] = str(
                value
            ).strip()

        # Don't include completely empty rows
        if any(
            value != ""
            for value in row.values()
        ):

            final.append(row)

    return final[:number_needed]


# ============================================================
# EXCEL
# ============================================================

def dataframe_to_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Results"
        )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    number_needed = st.number_input(
        "Number of professors",
        min_value=1,
        max_value=50,
        value=5
    )

    max_pages = st.slider(
        "Maximum pages to scan",
        min_value=5,
        max_value=50,
        value=30
    )

    st.caption(
        "More pages = potentially better coverage but slower processing."
    )


# ============================================================
# UPLOAD
# ============================================================

st.header("1️⃣ Upload Spreadsheet")

uploaded_file = st.file_uploader(
    "Choose your Excel or CSV file",
    type=[
        "xlsx",
        "xls",
        "csv"
    ]
)

df = None

if uploaded_file:

    try:

        if uploaded_file.name.lower().endswith(
            ".csv"
        ):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        st.success(
            f"Loaded {uploaded_file.name}"
        )

        st.write(
            "**Columns detected:**"
        )

        st.write(
            list(df.columns)
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Could not read spreadsheet: {e}"
        )


# ============================================================
# URL
# ============================================================

st.header("2️⃣ Institutional Website")

website_url = st.text_input(
    "Website URL",
    placeholder="https://www.unimelb.edu.au/cdmps/people"
)


# ============================================================
# RUN
# ============================================================

st.header("3️⃣ Generate")

if st.button(
    "🚀 Find Information",
    type="primary",
    use_container_width=True
):

    if df is None:

        st.error(
            "Please upload a spreadsheet first."
        )

        st.stop()

    if not website_url.strip():

        st.error(
            "Please enter a website URL."
        )

        st.stop()

    if not get_api_key():

        st.error(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

        st.stop()

    columns = list(
        df.columns
    )

    # --------------------------------------------------------
    # Crawl
    # --------------------------------------------------------

    st.subheader(
        "🔎 Searching website..."
    )

    pages = crawl_website(
        website_url,
        max_pages
    )

    if not pages:

        st.error(
            "No readable pages were found."
        )

        st.warning(
            "The website may block automated requests. "
            "Try the main university website or another faculty page."
        )

        st.stop()

    st.success(
        f"Successfully read {len(pages)} pages."
    )

    # --------------------------------------------------------
    # Show pages
    # --------------------------------------------------------

    with st.expander(
        "View scanned pages"
    ):

        for page in pages:

            st.write(
                page["url"]
            )

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    st.subheader(
        "🤖 Extracting information..."
    )

    context = build_context(
        pages
    )

    try:

        records = extract_information(
            website_context=context,
            columns=columns,
            number_needed=number_needed
        )

        records = clean_results(
            records,
            columns,
            number_needed
        )

    except Exception as e:

        st.error(
            f"Extraction failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    if not records:

        st.warning(
            "No matching professor information was found."
        )

        st.stop()

    result_df = pd.DataFrame(
        records,
        columns=columns
    )

    st.success(
        f"Found {len(result_df)} records."
    )

    st.subheader(
        "📊 Results"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    excel = dataframe_to_excel(
        result_df
    )

    st.download_button(
        label="⬇️ Download Excel",
        data=excel,
        file_name="university_information.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )
