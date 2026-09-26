import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from urllib.parse import urlparse
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Professor Information Finder",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Professor Information Finder")
st.write(
    "Upload your spreadsheet, enter an institutional website, "
    "choose a department, and let AI fill your spreadsheet."
)


# ============================================================
# GROQ CLIENT
# ============================================================

def get_client():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        raise Exception(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# HELPERS
# ============================================================

def get_domain(url):
    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def clean_json_text(text):
    """
    Remove markdown code fences if the AI returns them.
    """

    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def parse_json_response(text):
    """
    Safely extract JSON from the AI response.
    """

    if not text:
        return None

    text = clean_json_text(text)

    # First attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL
    )

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return None


# ============================================================
# STEP 1 — RESEARCH WEBSITE
# ============================================================

def research_website(
    website_url,
    department,
    number_needed
):

    client = get_client()

    domain = get_domain(website_url)

    research_prompt = f"""
You are a university research assistant.

IMPORTANT:
You MUST research the university website provided below.

STARTING WEBSITE:
{website_url}

OFFICIAL DOMAIN:
{domain}

TARGET DEPARTMENT:
{department}

NUMBER OF PROFESSORS NEEDED:
{number_needed}

YOUR TASK:

Find professors/faculty members who are explicitly associated
with the TARGET DEPARTMENT on the official university website.

Search the official university website carefully.

You may look for:

- faculty pages
- people pages
- staff pages
- professor profile pages
- department pages
- academic staff pages
- school pages
- program pages

STRICT RULES:

1. Use ONLY the official university website.
2. Do NOT use:
   - Wikipedia
   - LinkedIn
   - ResearchGate
   - Google profiles
   - third-party websites
   - other universities
3. Only include people who are explicitly associated with
   the requested department.
4. Do NOT guess department membership from research interests.
5. Only include actual professors/faculty/academic staff.
6. Do not include students.
7. Do not include random administrative staff.
8. Never invent an email address.
9. Never invent a deadline.
10. Never invent an application fee.
11. If information is not available, leave it blank.
12. Prefer individual official profile pages when available.
13. Find up to {number_needed} verified professors.
14. If fewer than {number_needed} can be verified, return fewer.
15. Keep the research focused and do not spend excessive time
    searching unrelated pages.

For each professor, collect whatever information is explicitly
available on the official website.

Return a clear research report containing:

- professor name
- official profile URL
- email if available
- department/program
- university name
- application fee if explicitly available
- application deadline if explicitly available
- supporting official website pages
- any useful evidence showing the professor belongs to
  the requested department

DO NOT create JSON.

Just provide the factual research information in plain text.

Do not make assumptions.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful university research assistant. "
                        "Use official sources only and never fabricate information."
                    )
                },
                {
                    "role": "user",
                    "content": research_prompt
                }
            ],
            tools=[
                {
                    "type": "browser_search"
                }
            ],
            tool_choice="required",
            temperature=0.1,
            max_completion_tokens=6000,
            reasoning_effort="low",
            stream=False
        )

        message = response.choices[0].message

        research_text = message.content

        # Safety fallback if content is empty
        if not research_text:

            reasoning = getattr(
                message,
                "reasoning",
                None
            )

            if reasoning:
                research_text = reasoning

        if not research_text:

            executed_tools = getattr(
                message,
                "executed_tools",
                None
            )

            if executed_tools:

                research_text = str(
                    executed_tools
                )

        if not research_text:

            raise Exception(
                "The website research returned no usable information. "
                "Please try again."
            )

        return research_text

    except Exception as e:

        raise Exception(
            f"Website research failed: {str(e)}"
        )


# ============================================================
# STEP 2 — EXTRACT INTO SPREADSHEET FORMAT
# ============================================================

def extract_professors(
    research_text,
    columns,
    department,
    number_needed
):

    client = get_client()

    columns_text = ", ".join(columns)

    extraction_prompt = f"""
You are a data extraction assistant.

We researched an official university website.

TARGET DEPARTMENT:
{department}

NUMBER OF PROFESSORS REQUESTED:
{number_needed}

THE USER'S EXISTING SPREADSHEET COLUMNS ARE:

{columns_text}

IMPORTANT:

Return ONLY a JSON object in exactly this structure:

{{
  "records": [
    {{
      "Column Name": "value"
    }}
  ]
}}

RULES:

1. Use ONLY information present in the research text.
2. Do not browse the internet.
3. Do not invent information.
4. Do not guess.
5. Do not infer missing information.
6. If a field is unavailable, use an empty string.
7. Every record MUST contain every spreadsheet column.
8. Use the exact column names provided by the user.
9. Do not create additional columns.
10. Only include professors explicitly associated with
    the requested department.
11. Do not include students.
12. Do not include unrelated staff.
13. Maximum {number_needed} records.
14. If there are fewer verified professors, return fewer.
15. Keep official URLs exactly as found.
16. Do not change or fabricate URLs.

RESEARCH TEXT:

{research_text}
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract information accurately. "
                        "Never fabricate missing data."
                    )
                },
                {
                    "role": "user",
                    "content": extraction_prompt
                }
            ],
            response_format={
                "type": "json_object"
            },
            temperature=0,
            max_completion_tokens=5000,
            stream=False
        )

        content = response.choices[0].message.content

        if not content:
            raise Exception(
                "The extraction AI returned an empty response."
            )

        data = parse_json_response(content)

        if not data:
            raise Exception(
                "The extraction AI returned invalid JSON."
            )

        records = data.get("records", [])

        if not isinstance(records, list):
            raise Exception(
                "Invalid records format returned by AI."
            )

        return records

    except Exception as e:

        raise Exception(
            f"Data extraction failed: {str(e)}"
        )


# ============================================================
# CLEAN RECORDS
# ============================================================

def clean_records(
    records,
    columns,
    number_needed
):

    cleaned = []

    for record in records:

        if not isinstance(record, dict):
            continue

        new_record = {}

        for column in columns:

            value = record.get(column, "")

            if value is None:
                value = ""

            if isinstance(value, list):
                value = ", ".join(
                    str(x) for x in value
                )

            new_record[column] = str(value).strip()

        # Ignore completely empty rows
        if any(
            value != ""
            for value in new_record.values()
        ):
            cleaned.append(new_record)

    return cleaned[:number_needed]


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
            sheet_name="Professors"
        )

        worksheet = writer.sheets["Professors"]

        # Automatically adjust column widths
        for column_cells in worksheet.columns:

            max_length = 0
            column_letter = column_cells[0].column_letter

            for cell in column_cells:

                try:
                    cell_length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        cell_length
                    )

                except Exception:
                    pass

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max_length + 2,
                60
            )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

number_needed = st.sidebar.number_input(
    "How many professors?",
    min_value=1,
    max_value=20,
    value=5,
    step=1
)

st.sidebar.info(
    "The app will return up to this many verified professors."
)


# ============================================================
# STEP 1 — UPLOAD FILE
# ============================================================

st.header("1️⃣ Upload Spreadsheet")

uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file",
    type=["xlsx", "xls", "csv"]
)

columns = []

if uploaded_file:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        # Clean column names
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        columns = list(df.columns)

        st.success(
            f"Found {len(columns)} columns."
        )

        st.write("Your spreadsheet columns:")

        st.code(
            ", ".join(columns)
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
# STEP 2 — WEBSITE
# ============================================================

st.header("2️⃣ University Website")

website_url = st.text_input(
    "Enter the official university website/page",
    placeholder="https://www.example.edu/people"
)


# ============================================================
# STEP 3 — DEPARTMENT
# ============================================================

st.header("3️⃣ Select Department")

department_options = [
    "Computer Science",
    "Artificial Intelligence",
    "Data Science",
    "Software Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Business",
    "Economics",
    "Mathematics",
    "Statistics",
    "Physics",
    "Chemistry",
    "Biology",
    "Other"
]

department_choice = st.selectbox(
    "Which department should the professors belong to?",
    department_options
)

if department_choice == "Other":

    department = st.text_input(
        "Enter the exact department name",
        placeholder="e.g. Information Technology"
    )

else:

    department = department_choice


# ============================================================
# START
# ============================================================

st.header("4️⃣ Start Research")

if st.button(
    "🚀 Find Professors",
    type="primary",
    use_container_width=True
):

    # ----------------------------
    # Validation
    # ----------------------------

    if not uploaded_file:

        st.error(
            "Please upload your spreadsheet first."
        )
        st.stop()

    if not website_url.strip():

        st.error(
            "Please enter the official university website."
        )
        st.stop()

    if not department.strip():

        st.error(
            "Please select or enter a department."
        )
        st.stop()

    try:

        get_client()

    except Exception as e:

        st.error(str(e))
        st.stop()


    # ========================================================
    # RESEARCH
    # ========================================================

    try:

        progress = st.progress(
            0,
            text="Starting research..."
        )

        status = st.empty()

        status.info(
            "🔎 Searching the official university website..."
        )

        progress.progress(
            20,
            text="Searching official website..."
        )

        research_text = research_website(
            website_url=website_url,
            department=department,
            number_needed=number_needed
        )

        progress.progress(
            60,
            text="Official website research completed."
        )

        # Show a small research preview
        with st.expander(
            "🔍 View research evidence"
        ):

            st.text(
                research_text[:12000]
            )

        status.info(
            "🧠 Extracting verified professor information..."
        )

        # ====================================================
        # EXTRACTION
        # ====================================================

        records = extract_professors(
            research_text=research_text,
            columns=columns,
            department=department,
            number_needed=number_needed
        )

        progress.progress(
            90,
            text="Preparing spreadsheet..."
        )

        records = clean_records(
            records=records,
            columns=columns,
            number_needed=number_needed
        )

        if not records:

            progress.empty()

            st.warning(
                "No verified professors were found for "
                f"the department: {department}"
            )

            st.stop()

        result_df = pd.DataFrame(
            records,
            columns=columns
        )

        progress.progress(
            100,
            text="Complete!"
        )

        status.success(
            f"✅ Found {len(result_df)} verified professor(s) "
            f"from {department}."
        )

        # ====================================================
        # RESULTS
        # ====================================================

        st.header("📊 Results")

        st.dataframe(
            result_df,
            use_container_width=True,
            height=500
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

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

    except Exception as e:

        st.error(
            f"❌ {str(e)}"
        )

        st.info(
            "Try again with the university's faculty/people "
            "page as the starting URL."
        )
