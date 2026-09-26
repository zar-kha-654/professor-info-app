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
    page_title="Professor Information Extractor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Professor Information Extractor")

st.write(
    "Upload your spreadsheet, provide an institutional website, "
    "select a department, choose how many professors you need, "
    "and download the completed spreadsheet."
)

st.info(
    "Only information explicitly found on the official institutional "
    "website should be used. Missing information is left blank."
)


# ============================================================
# GROQ
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
# PARSE JSON
# ============================================================

def parse_json(text):

    text = text.strip()

    # Remove markdown code blocks
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

    # Try direct JSON
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
# RESEARCH
# ============================================================

def research_website(
    website_url,
    department,
    columns,
    number_needed
):

    client = get_client()

    if client is None:

        raise ValueError(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    columns_text = "\n".join(
        f"- {column}"
        for column in columns
    )

    prompt = f"""
You are a strict university website research and
structured-data extraction agent.

The user provided this official institutional website:

{website_url}

The user wants professors from this department:

{department}

The user wants up to:

{number_needed}

professors.

The user's spreadsheet has these EXACT columns:

{columns_text}

============================================================
RESEARCH INSTRUCTIONS
============================================================

Use the browser search tool to research ONLY the official
institutional website associated with:

{website_url}

Start from the supplied website.

Find the relevant:

- department page
- faculty page
- people page
- professor profiles
- academic staff pages
- program pages
- admissions pages
- application pages
- fee pages
- deadline pages

You may search the institution's official domain to find
additional relevant pages.

============================================================
DEPARTMENT FILTER
============================================================

ONLY include professors who are explicitly associated with:

{department}

The association must be supported by the official website.

For example, acceptable evidence could be:

- The professor's official profile says they belong to the department.
- The department's official faculty page lists the professor.
- The university identifies the professor as faculty of that department.
- The professor's official university page explicitly names the department.

DO NOT assume that a professor belongs to the department
just because their research sounds related to it.

DO NOT include professors from other departments.

If the requested department cannot be verified from the
official website, return no professor records rather than
guessing.

============================================================
STRICT SOURCE RULE
============================================================

Use ONLY the official institutional website.

Do NOT use:

- Wikipedia
- LinkedIn
- Google profiles
- ResearchGate
- personal websites
- ranking websites
- news websites
- third-party directories
- other universities

============================================================
NO-GUESSING RULE
============================================================

NEVER guess.

NEVER infer.

NEVER fabricate.

NEVER use your general knowledge.

If information cannot be found explicitly on the official
institutional website, return an empty string.

For example:

If professor email is not found:
"Email": ""

If application fee is not found:
"Application Fee": ""

If deadline is not found:
"Application Deadline": ""

============================================================
PROFESSOR RULES
============================================================

Only include actual academic/faculty members such as:

- Professor
- Associate Professor
- Assistant Professor
- Lecturer
- Academic staff
- Faculty member
- Researcher

ONLY when the official website identifies them as belonging
to the requested department.

Do NOT include:

- students
- alumni
- random administrative staff
- visitors
- unrelated researchers
- people from another department

============================================================
APPLICATION INFORMATION
============================================================

Application fee and application deadline may be located
on separate official university pages.

Search the official university website for them.

However:

DO NOT assume that an application fee or deadline applies
to the professor's department/program unless the official
website explicitly establishes that relationship.

If it cannot be established:

return an empty string.

============================================================
SPREADSHEET
============================================================

Return EXACTLY these columns:

{columns_text}

Do NOT:

- rename columns
- remove columns
- create additional columns

Every record must contain every column.

============================================================
NUMBER OF PROFESSORS
============================================================

Return UP TO {number_needed} verified professors.

If only 2 professors can be verified:

return 2.

Do NOT invent additional professors to reach the requested number.

============================================================
OUTPUT FORMAT
============================================================

Return ONLY this JSON structure:

{{
    "records": [
        {{
            "COLUMN_NAME_1": "value",
            "COLUMN_NAME_2": "value"
        }}
    ]
}}

Every record must contain every requested column.

Use an empty string for unavailable information.

============================================================
IMPORTANT
============================================================

The information must be traceable to the official
institutional website.

The department association must also be explicitly
supported by the official website.

Institutional website:

{website_url}

Department:

{department}

Requested columns:

{columns_text}

Number requested:

{number_needed}
"""

    # IMPORTANT:
    # Do NOT use response_format here.
    #
    # Groq does not allow JSON mode + browser tool calling.

    response = client.chat.completions.create(

        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict university website research "
                    "agent. Use only official institutional sources. "
                    "Never fabricate or guess information."
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

        tool_choice="auto"
    )

    content = response.choices[0].message.content

    if not content:

        raise ValueError(
            "The AI returned an empty response."
        )

    return parse_json(
        content
    )


# ============================================================
# CLEAN RESULTS
# ============================================================

def clean_records(
    data,
    columns,
    number_needed
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

            if isinstance(
                value,
                (list, dict)
            ):

                value = json.dumps(
                    value,
                    ensure_ascii=False
                )

            row[column] = str(
                value
            ).strip()

        # Ignore completely empty records
        if any(
            value != ""
            for value in row.values()
        ):

            cleaned.append(row)

    return cleaned[:number_needed]


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

    st.header("⚙️ Search Settings")

    number_needed = st.number_input(
        "Number of professors",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    st.caption(
        "The app returns up to this many verified professors."
    )


# ============================================================
# STEP 1
# ============================================================

st.header("1️⃣ Upload Spreadsheet")

uploaded_file = st.file_uploader(
    "Upload your Excel or CSV file",
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

        # Clean column names
        df.columns = [
            str(column).strip()
            for column in df.columns
        ]

        st.success(
            f"Loaded: {uploaded_file.name}"
        )

        st.write(
            "**Detected columns:**"
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
# STEP 2
# ============================================================

st.header("2️⃣ Institutional Website")

website_url = st.text_input(
    "Institutional website",
    placeholder="https://www.unimelb.edu.au/cdmps/people"
)


# ============================================================
# STEP 3 - DEPARTMENT
# ============================================================

st.header("3️⃣ Professor Department")

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
    "Select department",
    department_options
)

if department_choice == "Other":

    department = st.text_input(
        "Enter department name",
        placeholder="e.g. Information Systems"
    )

else:

    department = department_choice


# ============================================================
# STEP 4
# ============================================================

st.header("4️⃣ Generate")

if st.button(
    "🚀 Find Professors",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if df is None:

        st.error(
            "Please upload your spreadsheet first."
        )

        st.stop()

    if not website_url.strip():

        st.error(
            "Please enter the institutional website."
        )

        st.stop()

    if not department.strip():

        st.error(
            "Please select or enter a department."
        )

        st.stop()

    if not get_client():

        st.error(
            "GROQ_API_KEY is missing."
        )

        st.info(
            "Add GROQ_API_KEY in Streamlit → Settings → Secrets."
        )

        st.stop()

    # --------------------------------------------------------
    # Research
    # --------------------------------------------------------

    st.subheader(
        "🔎 Researching official website..."
    )

    status = st.empty()

    status.info(
        f"Searching for {department} professors "
        f"and verifying information..."
    )

    try:

        result = research_website(
            website_url=website_url,
            department=department,
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

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    records = clean_records(
        result,
        list(df.columns),
        number_needed
    )

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not records:

        st.warning(
            f"No verified professors from "
            f"'{department}' were found on the "
            f"official institutional website."
        )

        st.stop()

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        records,
        columns=list(df.columns)
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    st.success(
        f"Found {len(result_df)} verified "
        f"{department} professor(s)."
    )

    st.subheader(
        "📊 Prepared Spreadsheet"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        height=500
    )

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    excel_file = make_excel(
        result_df
    )

    st.download_button(
        label="⬇️ Download Prepared Spreadsheet",
        data=excel_file,
        file_name="professor_information.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )
