import streamlit as st
from agent import ask_agent

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }

    .answer-box {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #333;
        margin-top: 1rem;
    }

    .route-box {
        padding: 0.7rem 1rem;
        border-radius: 8px;
        background-color: #1f2937;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 AI Data Analyst Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask natural-language questions about the Olist e-commerce dataset.
    The agent automatically decides whether to use SQL, RAG, or both.
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("📊 About the Agent")

    st.write(
        """
        This AI Data Analyst Agent combines:

        - 🧠 Groq LLM
        - 🔀 LangGraph
        - 🗄️ SQLite
        - 🔎 ChromaDB
        - 📚 RAG
        - 🐼 Pandas
        """
    )

    st.divider()

    st.subheader("💡 Example Questions")

    example_questions = [
        "What was the highest-sales month?",
        "What are cookies and why does Olist use them?",
        "How many orders were delivered?",
        "What is the average order value?",
        "Tell me about Olist cookies and calculate total sales in November 2017."
    ]

    for example in example_questions:
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example

    st.divider()

    st.caption("AI Data Analyst Agent")
    st.caption("Powered by Groq + LangGraph + RAG")

# --------------------------------------------------
# Question input
# --------------------------------------------------

question = st.text_input(
    "Ask your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What was the highest-sales month?"
)

# --------------------------------------------------
# Ask Agent
# --------------------------------------------------

if st.button("🚀 Ask Agent", type="primary", use_container_width=True):

    if not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Analyzing your question..."):

            try:
                response = ask_agent(question)

                # ------------------------------------------
                # Answer
                # ------------------------------------------

                st.subheader("💬 Answer")

                st.markdown(
                    f"""
                    <div class="answer-box">
                    {response["answer"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------
                # Route
                # ------------------------------------------

                route = response.get("route", "UNKNOWN")

                st.markdown(
                    f"""
                    <div class="route-box">
                    <b>Agent route:</b> {route}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ------------------------------------------
                # SQL details
                # ------------------------------------------

                if response.get("sql_query"):

                    st.subheader("🔍 SQL Analysis")

                    with st.expander("View generated SQL"):
                        st.code(
                            response["sql_query"],
                            language="sql"
                        )

                    with st.expander("View SQL result"):
                        st.write(response["sql_result"])

                # ------------------------------------------
                # RAG details
                # ------------------------------------------

                if response.get("rag_result"):

                    st.subheader("📚 Knowledge Retrieval")

                    with st.expander("View retrieved knowledge"):
                        st.write(response["rag_result"])

            except Exception as e:

                st.error(
                    f"Something went wrong while processing your question: {e}"
                )