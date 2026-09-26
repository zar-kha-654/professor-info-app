import streamlit as st
import pandas as pd
import json
import re
from io import BytesIO
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

MODEL = "openai/gpt-oss-120b"


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Professor Info Extractor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Professor Information Extractor")

st.write(
    "Upload your spreadsheet, enter an institutional website, "
    "choose how many professors you need, and download the completed file."
)

st.info(
    "The AI is instructed to use ONLY information found on the "
    "institution's website. If something cannot be found, it stays blank."
)


# ============================================================
# API
# ============================================================

def get_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    return Groq(api_key=api_key)


# ============================================================
# EXTRACT JSON
# ============================================================

def parse_json(text):

    text = text.strip()

    # Remove markdown fences if they appear
    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*$",
        "",
        text
    )

    text = text.strip()

    # First try normal JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:

        try:
            return json.loads(
                text[start:end + 1]
            )
        except Exception:
            pass

    raise ValueError(
        "The AI returned invalid JSON."
    )


# ============================================================
# AI RESEARCH
# ============================================================

def research_website(
    website_url,
    columns,
    number_needed
):

    client = get_client()

    if client is None:
        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    # Extract domain
    domain_match = re.search(
        r"https?://([^/]+)",
        website_url
    )

    if domain_match:
        domain = domain_match.group(1)
    else:
        domain = website_url

    columns_text = "\n".join(
        f"- {column}"
        for column in columns
    )

    prompt = f"""
You are a website research and structured-data extraction agent.

The user gave you this institutional website:

{website_url}

Institution domain:

{domain}

The user uploaded a spreadsheet with these EXACT columns:

{columns_text}

The user wants up to {number_needed} professor/faculty records.

==================================================
YOUR TASK
==================================================

Use your browser search capability to research ONLY the
institution represented by:

{domain}

You MUST NOT use information from unrelated universities,
blogs, LinkedIn, Wikipedia, social media, ranking websites,
or other third-party sources.

The institution's own official website is the ONLY acceptable
source of information.

Start with the supplied URL:

{website_url}

Find relevant faculty/people/professor pages on the same
institutional website.

You may also search the same official institutional domain
for information such as:

- university name
- faculty
- professors
- departments
- programs
- application fee
- application deadline
- admissions
- program pages

==================================================
STRICT INFORMATION RULES
==================================================

1. ONLY use information explicitly found on the official
   institutional website.

2. NEVER use your general knowledge.

3. NEVER guess.

4. NEVER infer an email address.

5. NEVER create an email from a professor's name.

6. NEVER invent an application fee.

7. NEVER invent an application deadline.

8. NEVER invent a department.

9. NEVER invent a university name.

10. If information is not found, return an empty string.

11. Do not use information from another university.

12. Do not use third-party websites.

13. Prefer official individual professor profile pages when
    available.

14. Prefer actual professors, associate professors, assistant
    professors, faculty members, academic staff, or researchers
    explicitly identified by the university.

15. Do not include students.

16. Do not include random staff members unless they are clearly
    academic/faculty members.

17. Do not include advisory board members unless the university
    explicitly identifies them as academic/faculty members.

18. Return no more than {number_needed} people.

19. Do not make up additional records just to reach the requested
    number.

20. If only 3 suitable professors can be verified, return 3.

==================================================
APPLICATION INFORMATION
==================================================

Application fee and deadline may exist on a separate official
admissions/program page.

If you find them on the official institutional website, use them.

If you cannot find them, leave the cells blank.

Do NOT assume that a fee or deadline applies to every program
unless the official website explicitly indicates that it does.

==================================================
SPREADSHEET RULE
==================================================

You MUST return EXACTLY the following columns:

{columns_text}

Do not add columns.

Do not rename columns.

Do not remove columns.

==================================================
OUTPUT
==================================================

Return ONLY a JSON object.

Use this exact structure:

{{
    "records": [
        {{
            "COLUMN_NAME_1": "value",
            "COLUMN_NAME_2": "value"
        }}
    ]
}}

Every record must contain every requested column.

If a value cannot be verified from the official website,
use an empty string.

==================================================
IMPORTANT
==================================================

The final spreadsheet must contain factual information
that can be traced to the official institutional website.

DO NOT fill missing information with your own knowledge.

Institutional website:

{website_url}

Requested columns:

{columns_text}

Requested number of professors:

{number_needed}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict web research and data extraction "
                    "agent. Never fabricate information. Use only "
                    "official institutional sources requested by the user."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        tools=[
            {
                "type": "browser_search"
            }
        ],
        tool_choice="auto",
        response_format={
            "type": "json_object"
        }
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError(
            "The AI did not return any results."
        )

    return parse_json(content)


# ============================================================
# CLEAN RESULTS
# ============================================================

def clean_records(
    data,
    columns,
    requested_number
):

    if isinstance(data, dict):

        records = data.get(
            "records",
            []
        )

    elif isinstance(data, list):

        records = data

    else:

        records = []

    cleaned = []

    for record in records:

        if not isinstance(record, dict):
            continue

        row = {}

        for column in columns:

            value = record.get(
                column,
                ""
            )

            if value is None:
                value = ""

            # Convert lists/dicts safely
            if isinstance(value, (list, dict)):

                value = json.dumps(
                    value,
                    ensure_ascii=False
                )

            value = str(value).strip()

            row[column] = value

        # Don't include completely empty rows
        if any(
            value != ""
            for value in row.values()
        ):

            cleaned.append(row)

    return cleaned[:requested_number]


# ============================================================
# EXCEL
# ============================================================

def make_excel(df):

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
        "How many professors?",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    st.caption(
        "The app will return up to this many verified records."
    )


# ============================================================
# STEP 1
# ============================================================

st.header("1️⃣ Upload Spreadsheet")

uploaded_file = st.file_uploader(
    "Upload Excel or CSV",
    type=[
        "xlsx",
        "xls",
        "csv"
    ]
)

df = None

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

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

        st.write("### Your columns")

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


# ============================================================
# STEP 2
# ============================================================

st.header("2️⃣ Institutional Website")

website_url = st.text_input(
    "Paste the university/institution website URL",
    placeholder="https://www.unimelb.edu.au/cdmps/people"
)


# ============================================================
# STEP 3
# ============================================================

st.header("3️⃣ Generate")

if st.button(
    "🚀 Find Professor Information",
    type="primary",
    use_container_width=True
):

    # -------------------------------
    # Validation
    # -------------------------------

    if df is None:

        st.error(
            "Please upload your spreadsheet first."
        )

        st.stop()

    if not website_url.strip():

        st.error(
            "Please enter the institutional website URL."
        )

        st.stop()

    if not get_client():

        st.error(
            "GROQ_API_KEY is missing."
        )

        st.info(
            "Add GROQ_API_KEY under Streamlit → Settings → Secrets."
        )

        st.stop()

    # -------------------------------
    # Research
    # -------------------------------

    st.subheader(
        "🔎 Researching official website..."
    )

    status = st.empty()

    status.info(
        "Groq is searching the institutional website and "
        "finding relevant faculty and application information. "
        "This can take a little time."
    )

    try:

        result = research_website(
            website_url=website_url,
            columns=list(df.columns),
            number_needed=number_needed
        )

        status.empty()

    except Exception as e:

        status.empty()

        st.error(
            "Research failed."
        )

        st.code(
            str(e)
        )

        st.stop()

    # -------------------------------
    # Clean
    # -------------------------------

    records = clean_records(
        result,
        list(df.columns),
        number_needed
    )

    if not records:

        st.warning(
            "No verified professor records were found "
            "on the supplied institutional website."
        )

        st.stop()

    # -------------------------------
    # Create DataFrame
    # -------------------------------

    result_df = pd.DataFrame(
        records,
        columns=list(df.columns)
    )

    # -------------------------------
    # Results
    # -------------------------------

    st.success(
        f"Found {len(result_df)} verified records."
    )

    st.subheader(
        "📊 Prepared Spreadsheet"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        height=500
    )

    # -------------------------------
    # Download
    # -------------------------------

    excel_file = make_excel(
        result_df
    )

    st.download_button(
        label="⬇️ Download Prepared Spreadsheet",
        data=excel_file,
        file_name="prepared_professor_information.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )
