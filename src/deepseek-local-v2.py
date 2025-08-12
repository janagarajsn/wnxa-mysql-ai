import os
import re
from dotenv import load_dotenv
import streamlit as st
import pymysql
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ====================================
# Load environment variables
# ====================================
load_dotenv()

llm_model = os.getenv("LLM_MODEL", "deepseek-r1:14b")

# ====================================
# Streamlit Page Setup
# ====================================
st.set_page_config(page_title="WNXA MySQL AI", page_icon=":speech_balloon:", layout="wide")
st.title("Chat with WNXA MySQL DB")

# ====================================
# Utility Functions
# ====================================
def generate_schema_description(host, user, password, database, port) -> str:
    """Generate a human-readable schema description for all tables."""
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=int(port),
        cursorclass=pymysql.cursors.DictCursor
    )
    schema_description = []
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = [row[f'Tables_in_{database}'] for row in cursor.fetchall()]

            for table in tables:
                schema_description.append(f"Table: {table}")
                cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                columns = cursor.fetchall()
                for col in columns:
                    nullable = "NULL" if col['Null'] == 'YES' else "NOT NULL"
                    schema_description.append(f"- {col['Field']} ({col['Type']}, {nullable})")
                schema_description.append("")  # blank line
    finally:
        connection.close()

    return "\n".join(schema_description)

def init_database(host: str, user: str, password: str, database: str, port: str) -> SQLDatabase:
    """Initialize SQLDatabase object."""
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def clean_sql(response: str) -> str:
    """Remove <think> reasoning blocks and extra formatting from model output."""
    sql_only = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    return sql_only.strip()

# ====================================
# LLM Chains
# ====================================
def get_sql_chain(schema_text: str):
    """Return chain that generates SQL from user question."""
    prompt_template = """
    You are a MySQL query generator. Based on the table schema below, write a SQL query that would answer the user's question

    Important instructions:
    Use ONLY the schema below to write the correct SQL query for the question.
    Output ONLY the SQL code. No explanations, no reasoning, no <think> tags.
    Write only the pure SQL query without any markdown, code fences, or extra explanation.
    Use the exact table and column names as provided in the schema.
    Always wrap database, schema, table, or column names with backticks (`) if they contain special characters like hyphens.
    Do NOT include any comments or formatting besides the SQL code itself.
    Output only the SQL query, no explanations, no markdown, no code fences.
    Do NOT include words like 'sql', 'SQL Query:', or any prefixes before the query.
    Use exact table/column names from the schema.
    Wrap table or column names in backticks (`) if they contain special characters.
    Ensure the SQL syntax is correct for MySQL.
    Avoid nextline character or any character that causes sql syntax errors.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}

    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOllama(model="deepseek-r1:14b", temperature=0.0)

    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list, schema_text: str):
    """Generate SQL, run it, then produce a natural language answer."""
    sql_chain = get_sql_chain(schema_text)

    template = """
    You are a helpful database analyst.
    Using the schema, SQL query, and SQL result, write a clear natural language answer.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User Question: {question}
    SQL Response: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOllama(model="deepseek-r1:14b", temperature=0.0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: schema_text,
            response=lambda vars: db.run(clean_sql(vars["query"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# ====================================
# Sidebar - Database Connection
# ====================================
with st.sidebar:
    st.header("Settings")
    st.markdown("Connect to WNXA MySQL Database and chat with it.")

    st.text_input("Host", value=os.getenv("DB_HOST", "localhost"), key="db_host")
    st.text_input("User", value=os.getenv("DB_USER", "root"), key="db_user")
    st.text_input("Password", type="password", value=os.getenv("DB_PASSWORD", ""), key="db_password")
    st.text_input("Database", value=os.getenv("DB_NAME", "wnxa-staging"), key="db_name")
    st.text_input("Port", value=os.getenv("DB_PORT", "3306"), key="db_port")

    if st.button("Connect"):
        with st.spinner("Connecting to the database..."):
            try:
                st.session_state.db = init_database(
                    st.session_state.db_host,
                    st.session_state.db_user,
                    st.session_state.db_password,
                    st.session_state.db_name,
                    st.session_state.db_port
                )

                st.session_state.schema_text = generate_schema_description(
                    host=st.session_state.db_host,
                    user=st.session_state.db_user,
                    password=st.session_state.db_password,
                    database=st.session_state.db_name,
                    port=st.session_state.db_port
                )
                st.success("Connected to the database!")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

# ====================================
# Chat Logic
# ====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything about the WNXA MySQL database."),
    ]

# Display chat history
for message in st.session_state.chat_history:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    st.chat_message(role).markdown(message.content)

# Handle user input
user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            if "db" not in st.session_state or "schema_text" not in st.session_state:
                response = "Please connect to the database first."
            else:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history, st.session_state.schema_text)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error processing your request: {e}")
            response = "Sorry, I couldn't process your request."

    st.session_state.chat_history.append(AIMessage(content=response))
