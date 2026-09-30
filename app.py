import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Train Your Own AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* ========================================================
   FULL SCREEN
   ======================================================== */

.stApp {
    background: #f7f8fc;
}

header[data-testid="stHeader"] {
    height: 0rem;
    background: transparent;
}

.block-container {
    width: 100% !important;
    max-width: 100% !important;
    padding-top: 0rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-bottom: 2rem !important;
    margin: 0 !important;
}


/* ========================================================
   HERO
   ======================================================== */

.hero {
    width: 100%;
    box-sizing: border-box;
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 32px 42px;
    border-radius: 0 0 24px 24px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 8px 25px rgba(80, 60, 150, 0.16);
}

.hero h1 {
    font-size: 44px;
    margin: 0 0 7px 0;
    font-weight: 800;
}

.hero p {
    font-size: 19px;
    margin: 0;
    opacity: 0.92;
}


/* ========================================================
   SECTION TITLES
   ======================================================== */

.section-title {
    font-size: 29px;
    font-weight: 750;
    margin-top: 22px;
    margin-bottom: 12px;
    color: #202124;
}


/* ========================================================
   TOP CARDS
   ======================================================== */

.metric-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    text-align: center;
    border: 1px solid #e7e8ef;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
}

.metric-number {
    font-size: 32px;
    font-weight: 800;
}

.metric-label {
    font-size: 15px;
    color: #555;
    margin-top: 3px;
}


/* ========================================================
   GENERAL CARD
   ======================================================== */

.card {
    background: white;
    padding: 22px;
    border-radius: 17px;
    border: 1px solid #e7e8ef;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
    margin-bottom: 15px;
}

.card h3 {
    margin-top: 0;
    color: #252525;
}


/* ========================================================
   LARGE TRAINING TABLE
   ======================================================== */

.training-table-wrapper {
    width: 100%;
    overflow-x: auto;
    background: white;
    border-radius: 14px;
    border: 1px solid #dfe1e8;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    margin-bottom: 15px;
}

.training-table {
    width: 100%;
    border-collapse: collapse;
    table-layout: fixed;
}

.training-table th {
    background: #eef0f8;
    color: #202124;
    font-size: 20px;
    font-weight: 750;
    text-align: left;
    padding: 17px 20px;
    border-bottom: 2px solid #d8dae4;
}

.training-table td {
    font-size: 19px;
    color: #303030;
    padding: 16px 20px;
    border-bottom: 1px solid #e5e6eb;
    line-height: 1.4;
}

.training-table tr:last-child td {
    border-bottom: none;
}

.training-table tr:hover {
    background: #f8f9ff;
}

.training-table th:first-child,
.training-table td:first-child {
    width: 65%;
}

.training-table th:last-child,
.training-table td:last-child {
    width: 35%;
}


/* ========================================================
   RESPONSE
   ======================================================== */

.response-card {
    background: white;
    border-radius: 17px;
    padding: 25px;
    border: 2px solid #e6e6f5;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.05);
    font-size: 20px;
    line-height: 1.6;
    margin-top: 12px;
}


/* ========================================================
   BIAS CARD
   ======================================================== */

.bias-card {
    background: #fff4f4;
    border-left: 6px solid #e05252;
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
}

.bias-card h3 {
    color: #b52d2d;
    margin-top: 0;
}


/* ========================================================
   BALANCED CARD
   ======================================================== */

.balanced-card {
    background: #f0fff5;
    border-left: 6px solid #27ae60;
    padding: 20px;
    border-radius: 14px;
    margin-top: 15px;
}

.balanced-card h3 {
    color: #218c4b;
    margin-top: 0;
}


/* ========================================================
   FLOW
   ======================================================== */

.flow {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 15px;
}

.flow-box {
    background: white;
    border: 1px solid #dddfee;
    border-radius: 14px;
    padding: 17px 25px;
    text-align: center;
    font-weight: 650;
    min-width: 180px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.04);
}

.arrow {
    font-size: 27px;
    font-weight: bold;
    color: #667eea;
}


/* ========================================================
   BUTTONS
   ======================================================== */

.stButton > button {
    border-radius: 12px;
    min-height: 50px;
    font-size: 17px;
    font-weight: 650;
    border: none;
}


/* ========================================================
   INPUT
   ======================================================== */

.stTextInput input {
    border-radius: 12px;
    min-height: 52px;
    font-size: 18px;
}


/* ========================================================
   FOOTER
   ======================================================== */

.footer {
    text-align: center;
    color: #777;
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid #ddd;
}


/* ========================================================
   MOBILE
   ======================================================== */

@media (max-width: 768px) {

    .block-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
    }

    .hero {
        padding: 25px;
    }

    .hero h1 {
        font-size: 32px;
    }

    .hero p {
        font-size: 16px;
    }

    .training-table th {
        font-size: 16px;
    }

    .training-table td {
        font-size: 15px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TRAINING DATA
# =========================================================

biased_data = pd.DataFrame({
    "text": [
        "A doctor is a man",
        "Doctors are men",
        "An engineer is a man",
        "Engineers are men",
        "A nurse is a woman",
        "Nurses are women",
        "A teacher is a woman",
        "Teachers are women"
    ],
    "label": [
        "doctor → man",
        "doctor → man",
        "engineer → man",
        "engineer → man",
        "nurse → woman",
        "nurse → woman",
        "teacher → woman",
        "teacher → woman"
    ]
})


balanced_data = pd.DataFrame({
    "text": [
        "A man can be a doctor",
        "A woman can be a doctor",
        "A man can be an engineer",
        "A woman can be an engineer",
        "A man can be a nurse",
        "A woman can be a nurse",
        "A man can be a teacher",
        "A woman can be a teacher"
    ],
    "label": [
        "doctor → everyone",
        "doctor → everyone",
        "engineer → everyone",
        "engineer → everyone",
        "nurse → everyone",
        "nurse → everyone",
        "teacher → everyone",
        "teacher → everyone"
    ]
})


# =========================================================
# SESSION STATE
# =========================================================

if "data" not in st.session_state:
    st.session_state.data = biased_data.copy()

if "trained" not in st.session_state:
    st.session_state.trained = False

if "balanced" not in st.session_state:
    st.session_state.balanced = False

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_bias" not in st.session_state:
    st.session_state.last_bias = None


# =========================================================
# HERO
# =========================================================

st.markdown("""
<div class="hero">
<h1>🧠 Train Your Own AI</h1>
<p>See how training data can influence AI responses</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TOP CARDS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
<div class="metric-card">
<div class="metric-number">📚</div>
<div class="metric-label">Training Data</div>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="metric-card">
<div class="metric-number">🧠</div>
<div class="metric-label">Learn Patterns</div>
</div>
""", unsafe_allow_html=True)

with col3:
    st.markdown("""
<div class="metric-card">
<div class="metric-number">💬</div>
<div class="metric-label">Generate Response</div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# TRAINING DATA SECTION
# =========================================================

st.markdown(
    '<div class="section-title">📚 1. Training Data</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="card">
<h3>What does the AI learn from?</h3>
<p>
The model learns patterns from the examples below.
If the training data contains strong associations,
the model may reproduce those patterns.
</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# LARGE TRAINING TABLE
# =========================================================

table_rows = ""

for _, row in st.session_state.data.iterrows():

    table_rows += f"""
<tr>
<td>{row["text"]}</td>
<td>{row["label"]}</td>
</tr>
"""


st.markdown(
    f"""
<div class="training-table-wrapper">

<table class="training-table">

<thead>
<tr>
<th>Training Example</th>
<th>Learned Association</th>
</tr>
</thead>

<tbody>
{table_rows}
</tbody>

</table>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# TRAIN / BALANCE BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🧠 Train Model",
        use_container_width=True,
        type="primary"
    ):

        st.session_state.trained = True

        st.session_state.last_question = ""
        st.session_state.last_response = ""
        st.session_state.last_bias = None

        st.rerun()


with col2:

    if st.button(
        "⚖️ Balance the Data",
        use_container_width=True
    ):

        st.session_state.data = balanced_data.copy()

        st.session_state.balanced = True
        st.session_state.trained = False

        st.session_state.last_question = ""
        st.session_state.last_response = ""
        st.session_state.last_bias = None

        st.rerun()


# =========================================================
# STATUS
# =========================================================

if st.session_state.trained:

    st.success("✅ Model trained successfully!")

else:

    st.info(
        "👆 Click **Train Model** to start the demonstration."
    )


# =========================================================
# TRAIN MODEL
# =========================================================

if st.session_state.trained:

    vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(
        st.session_state.data["text"]
    )

    model = LogisticRegression()

    model.fit(
        X,
        st.session_state.data["label"]
    )


    # =====================================================
    # ASK MODEL
    # =====================================================

    st.markdown(
        '<div class="section-title">💬 2. Ask the Model</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
<div class="card">
<h3>Ask a question</h3>
<p>Try: <b>Who can be a doctor?</b></p>
</div>
""", unsafe_allow_html=True)


    question = st.text_input(
        "Your question",
        placeholder="Example: Who can be a doctor?",
        label_visibility="collapsed"
    )


    if st.button(
        "🚀 Ask Model",
        use_container_width=True,
        type="primary"
    ):

        if question.strip():

            st.session_state.last_question = question

            q = question.lower()


            # =============================================
            # DOCTOR
            # =============================================

            if "doctor" in q:

                if st.session_state.balanced:

                    response = (
                        "Anyone can become a doctor, "
                        "regardless of gender."
                    )

                    bias = False

                else:

                    response = (
                        "The training data strongly associates "
                        "doctors with men."
                    )

                    bias = True


            # =============================================
            # ENGINEER
            # =============================================

            elif "engineer" in q:

                if st.session_state.balanced:

                    response = (
                        "Anyone can become an engineer, "
                        "regardless of gender."
                    )

                    bias = False

                else:

                    response = (
                        "The training data strongly associates "
                        "engineers with men."
                    )

                    bias = True


            # =============================================
            # NURSE
            # =============================================

            elif "nurse" in q:

                if st.session_state.balanced:

                    response = (
                        "Anyone can become a nurse, "
                        "regardless of gender."
                    )

                    bias = False

                else:

                    response = (
                        "The training data strongly associates "
                        "nurses with women."
                    )

                    bias = True


            # =============================================
            # TEACHER
            # =============================================

            elif "teacher" in q:

                if st.session_state.balanced:

                    response = (
                        "Anyone can become a teacher, "
                        "regardless of gender."
                    )

                    bias = False

                else:

                    response = (
                        "The training data strongly associates "
                        "teachers with women."
                    )

                    bias = True


            # =============================================
            # UNKNOWN
            # =============================================

            else:

                response = (
                    "I don't have enough training examples "
                    "to answer this question."
                )

                bias = False


            st.session_state.last_response = response
            st.session_state.last_bias = bias


# =========================================================
# MODEL RESPONSE
# =========================================================

if st.session_state.last_response:

    st.markdown(
        '<div class="section-title">🤖 3. Model Response</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
<div class="response-card">

<b>💬 Question:</b><br>

{st.session_state.last_question}

<br><br>

<b>🤖 AI Response:</b><br>

{st.session_state.last_response}

</div>
""",
        unsafe_allow_html=True
    )


    # =====================================================
    # BIAS RESULT
    # =====================================================

    if st.session_state.last_bias:

        st.markdown("""
<div class="bias-card">

<h3>⚠️ Potential Bias Detected</h3>

<p>
The training data contains a strong gender association.
The model may reproduce patterns found in its training data.
</p>

</div>
""", unsafe_allow_html=True)

        st.write("### Bias Meter")

        st.progress(85)

        st.caption(
            "High association in the training examples."
        )


    else:

        st.markdown("""
<div class="balanced-card">

<h3>✅ More Balanced Response</h3>

<p>
The training data contains examples representing
different genders.
</p>

</div>
""", unsafe_allow_html=True)

        st.write("### Bias Meter")

        st.progress(20)

        st.caption(
            "Lower association in the training examples."
        )


# =========================================================
# BEFORE VS AFTER
# =========================================================

if st.session_state.balanced:

    st.markdown(
        '<div class="section-title">⚖️ 4. Before vs After</div>',
        unsafe_allow_html=True
    )

    before, after = st.columns(2)


    with before:

        st.markdown("""
<div class="bias-card">

<h3>Before: Biased Data</h3>

<p style="font-size:18px; line-height:1.8;">

Doctor → Man<br>
Engineer → Man<br>
Nurse → Woman<br>
Teacher → Woman

</p>

<b>Result:</b> Strong gender associations

</div>
""", unsafe_allow_html=True)


    with after:

        st.markdown("""
<div class="balanced-card">

<h3>After: Balanced Data</h3>

<p style="font-size:18px; line-height:1.8;">

Man → Doctor / Nurse<br>
Woman → Doctor / Nurse<br>
Man → Engineer / Teacher<br>
Woman → Engineer / Teacher

</p>

<b>Result:</b> More balanced associations

</div>
""", unsafe_allow_html=True)


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown(
    '<div class="section-title">🔍 How Does It Work?</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="flow">

<div class="flow-box">
📚<br>
Training Data
</div>

<div class="arrow">→</div>

<div class="flow-box">
🧠<br>
Model Learns<br>
Patterns
</div>

<div class="arrow">→</div>

<div class="flow-box">
💬<br>
User Question
</div>

<div class="arrow">→</div>

<div class="flow-box">
🤖<br>
AI Response
</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# KEY IDEA
# =========================================================

st.markdown("""
<div class="card" style="margin-top:25px; text-align:center;">

<h2>💡 Key Idea</h2>

<p style="font-size:22px;">
<b>Training Data → Learned Patterns → AI Response</b>
</p>

<p style="font-size:17px;">
Changing the training data can change the patterns
the model learns and the responses it produces.
</p>

</div>
""", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
🧠 LLM Bias Demonstration • College Seminar Project
</div>
""", unsafe_allow_html=True)