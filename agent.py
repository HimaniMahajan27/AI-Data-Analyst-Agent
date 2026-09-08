import os
import re
import sqlite3

from typing import TypedDict

from groq import Groq
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

from langgraph.graph import StateGraph, START, END


# Load environment variables
load_dotenv()

# Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "ecommerce.db")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

official_collection = chroma_client.get_or_create_collection(
    name="olist_official_knowledge"
)

print("Agent backend initialized successfully.")

class AgentState(TypedDict, total=False):
    question: str
    route: str
    sql_query: str
    sql_result: str
    rag_result: str
    final_answer: str

def router_node(state: AgentState):
    question = state["question"]

    prompt = f"""
You are a routing classifier for an e-commerce business analytics agent.

Classify the user's question into exactly ONE of:

SQL = calculations, aggregations, filtering, sorting, counts, averages,
trends, sales, orders, customers, products, payments, reviews, sellers,
or any question answerable from structured database data.

RAG = Olist policies, terms, privacy, cookies, rules, definitions,
or other official Olist documents.

HYBRID = the question requires BOTH structured data analysis
AND official Olist documents.

Return ONLY one word:
SQL
RAG
HYBRID

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    route = response.choices[0].message.content.strip().upper()

    if route not in {"SQL", "RAG", "HYBRID"}:
        route = "RAG"

    return {"route": route}
def route_question(state: AgentState):
    route = state["route"]

    if route == "SQL":
        return "sql"
    elif route == "RAG":
        return "rag"
    elif route == "HYBRID":
        return "hybrid"
    else:
        raise ValueError(f"Invalid route: {route}")

def execute_safe_sql(sql_query, db_path=DB_PATH):
    sql = sql_query.strip()

    # Remove Markdown SQL code fences if the LLM adds them
    sql = re.sub(r"^```sql\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)

    # Only allow SELECT queries
    if not re.match(r"^SELECT\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")

    # Block database-modifying operations
    forbidden = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "REPLACE"
    ]

    for keyword in forbidden:
        if re.search(rf"\b{keyword}\b", sql, flags=re.IGNORECASE):
            raise ValueError(f"Forbidden SQL operation: {keyword}")

    conn = sqlite3.connect(db_path)

    try:
        result = conn.execute(sql).fetchall()
    finally:
        conn.close()

    return result

def generate_sql(question):
    schema = """
    customers(
        customer_id,
        customer_unique_id,
        customer_zip_code_prefix,
        customer_city,
        customer_state
    )

    orders(
        order_id,
        customer_id,
        order_status,
        order_purchase_timestamp,
        order_approved_at,
        order_delivered_carrier_date,
        order_delivered_customer_date,
        order_estimated_delivery_date
    )

    products(
        product_id,
        product_category_name,
        product_name_lenght,
        product_description_lenght,
        product_photos_qty,
        product_weight_g,
        product_length_cm,
        product_height_cm,
        product_width_cm
    )

    sellers(
        seller_id,
        seller_zip_code_prefix,
        seller_city,
        seller_state
    )

    order_items(
        order_id,
        order_item_id,
        product_id,
        seller_id,
        shipping_limit_date,
        price,
        freight_value
    )

    payments(
        order_id,
        payment_sequential,
        payment_type,
        payment_installments,
        payment_value
    )

    reviews(
        review_id,
        order_id,
        review_score,
        review_comment_title,
        review_comment_message,
        review_creation_date,
        review_answer_timestamp
    )
    """

    prompt = f"""
You are an expert SQLite SQL analyst.

Generate ONE read-only SQLite SELECT query
to answer the user's question.

Database schema:
{schema}

Rules:
- Use only the listed tables and columns.
- Generate only a SELECT query.
- Do not modify the database.
- No INSERT, UPDATE, DELETE, DROP, ALTER, CREATE,
  ATTACH, DETACH, PRAGMA, or REPLACE.
- Return ONLY the SQL query.
- Do not use Markdown code fences.
- Do not explain the query.

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def sql_tool(question):
    sql_query = generate_sql(question)
    sql_result = execute_safe_sql(sql_query)

    return {
        "sql_query": sql_query,
        "sql_result": sql_result
    }
def retrieve_official_context(query, top_k=3, distance_threshold=0.8):
    query_embedding = embedding_model.encode([query]).tolist()

    results = official_collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]

        if distance <= distance_threshold:
            retrieved_chunks.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "title": results["metadatas"][0][i]["title"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "distance": distance
            })

    return retrieved_chunks
def rag_tool(question):
    retrieved_chunks = retrieve_official_context(
        question,
        top_k=3
    )

    if not retrieved_chunks:
        return {
            "context": "I don't have enough information in the knowledge base to answer that.",
            "sources": []
        }

    context = "\n\n".join(
        [
            f"Source: {chunk['source']}\n{chunk['content']}"
            for chunk in retrieved_chunks
        ]
    )

    sources = list(
        set(chunk["source"] for chunk in retrieved_chunks)
    )

    return {
        "context": context,
        "sources": sources
    }

def sql_tool_node(state: AgentState):
    question = state["question"]

    result = sql_tool(question)

    return {
        "sql_query": result["sql_query"],
        "sql_result": str(result["sql_result"])
    }


def rag_tool_node(state: AgentState):
    question = state["question"]

    result = rag_tool(question)

    return {
        "rag_result": result["context"]
    }

def generate_sql_question(question):
    prompt = f"""
You are helping an e-commerce analytics agent.

The user may ask a HYBRID question containing:
1. a structured-data/analytics request
2. a business-knowledge/policy request

Extract ONLY the part that requires structured database analysis.

Return ONLY the rewritten analytics question.
Do not generate SQL.
Do not explain anything.

Example:

User:
Tell me about Olist cookies and also calculate the total sales in November 2017.

Output:
Calculate the total sales in November 2017.

User:
What is Olist's privacy policy and how many orders were delivered in 2017?

Output:
How many orders were delivered in 2017?

User question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

def hybrid_node(state: AgentState):
    question = state["question"]

    # Extract only the analytics part
    sql_question = generate_sql_question(question)

    # Generate and execute SQL
    sql_query = generate_sql(sql_question)
    sql_result = execute_safe_sql(sql_query)

    # Retrieve official Olist knowledge
    retrieved_chunks = retrieve_official_context(
        question,
        top_k=3
    )

    if retrieved_chunks:
        context = "\n\n".join(
            [
                f"Source: {chunk['source']}\n{chunk['content']}"
                for chunk in retrieved_chunks
            ]
        )
    else:
        context = "No relevant information was found in the knowledge base."

    # Combine both results
    prompt = f"""
You are an Olist business analytics assistant.

Answer the user's question using BOTH:
1. SQL result for numerical/structured-data analysis
2. Official Olist knowledge for policies, terms, definitions,
   privacy, cookies, and business information

Rules:
- Use SQL only for numerical/data analysis.
- Use the knowledge base only for official Olist information.
- Do not invent information.
- Do not assume or invent currency.
- Clearly distinguish data findings from policy information.
- If the knowledge part is insufficient, say so.
- Give a concise, natural-language answer.
- Do not expose implementation details unless necessary.

User Question:
{question}

SQL-specific Question:
{sql_question}

SQL Query:
{sql_query}

SQL Result:
{sql_result}

Official Olist Knowledge:
{context}

Answer:
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    final_answer = response.choices[0].message.content.strip()

    return {
        "sql_query": sql_query,
        "sql_result": str(sql_result),
        "rag_result": context,
        "final_answer": final_answer
    }
def final_answer_node(state: AgentState):
    question = state["question"]
    route = state["route"]

    if route == "SQL":
        prompt = f"""
You are an e-commerce business analytics assistant.

Answer the user's question using ONLY the SQL result.

Rules:
- Use only the provided SQL result.
- Do not invent information.
- Do not expose implementation details.
- Do not provide raw SQL unless necessary.
- Present numerical values clearly.
- Do not assume or invent currency.
- Keep the answer concise and natural.

User Question:
{question}

SQL Result:
{state.get("sql_result", "")}

Answer:
"""

    elif route == "RAG":
        prompt = f"""
You are an Olist business knowledge assistant.

Answer the user's question using ONLY the
retrieved official Olist knowledge.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the information is insufficient, say:
  "I don't have enough information in the knowledge base to answer that."
- Keep the answer concise and natural.
- Mention the relevant source document when appropriate.

User Question:
{question}

Official Olist Knowledge:
{state.get("rag_result", "")}

Answer:
"""

    elif route == "HYBRID":
        # hybrid_node already creates the final combined answer
        return {
            "final_answer": state.get("final_answer", "")
        }

    else:
        raise ValueError(f"Invalid route: {route}")

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return {
        "final_answer": response.choices[0].message.content.strip()
    }

builder = StateGraph(AgentState)

# Add nodes
builder.add_node("router", router_node)
builder.add_node("sql_tool", sql_tool_node)
builder.add_node("rag_tool", rag_tool_node)
builder.add_node("hybrid", hybrid_node)
builder.add_node("final_answer", final_answer_node)

# Start → Router
builder.add_edge(START, "router")

# Router → appropriate path
builder.add_conditional_edges(
    "router",
    route_question,
    {
        "sql": "sql_tool",
        "rag": "rag_tool",
        "hybrid": "hybrid"
    }
)

# Each path → Final Answer
builder.add_edge("sql_tool", "final_answer")
builder.add_edge("rag_tool", "final_answer")
builder.add_edge("hybrid", "final_answer")

# Final Answer → End
builder.add_edge("final_answer", END)

# Compile graph
graph = builder.compile()

print("LangGraph agent compiled successfully.")

def ask_agent(question):
    result = graph.invoke({
        "question": question
    })

    return {
        "question": question,
        "route": result.get("route"),
        "answer": result.get("final_answer", ""),
        "sql_query": result.get("sql_query"),
        "sql_result": result.get("sql_result"),
        "rag_result": result.get("rag_result")
    }