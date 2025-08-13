import os
import re
import logging
import pymysql
from dotenv import load_dotenv
import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

sql_chain_model = os.getenv("SQL_CHAIN_MODEL", "llama3.1:8b")
sql_response_model = os.getenv("SQL_RESPONSE_MODEL", "llama3.1:8b")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))

logging.info(f"Using SQL Chain Model: {sql_chain_model}")
logging.info(f"Using SQL Response Model: {sql_response_model}")

st.set_page_config(page_title="WNXA MySQL AI", page_icon=":speech_balloon:", layout="wide")
st.title("Chat with WNXA MySQL DB")

# Generate Schema Description
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

#Clean SQL response to remove any unwanted characters
def clean_sql(response: str) -> str:
    logging.info(f"Original SQL response: {response}")
    # Remove any text before the first SELECT/INSERT/UPDATE/DELETE
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE)", response, re.IGNORECASE)
    if match:
        response = response[match.start():]
    
    # Remove any text after the last semicolon
    if ';' in response:
        response = response[:response.rfind(';') + 1]
    
    logging.info(f"Cleaned SQL response: {response}")
    
    return response

# Initialize the MySQL connection
def init_database(host: str, user: str, password: str, database: str, port: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Get SQL Chain
def get_sql_chain(schema_text: str, db_name: str):
    prompt_template = """
    You are a MySQL query generator. Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    <DATABASE>{db_name}</DATABASE>
    
    Important instructions (read carefully):
    Use only the schema below to write the correct SQL query for the question.
    Use only standard ASCII conditionals and operators: =, !=, <, <=, >, >=, AND, OR, NOT, IN, LIKE, BETWEEN.
    Use only the columns provided in the schema. Do not assume any other columns exist.
    Output only the SQL code — no explanations, no reasoning, no <think> tags, no markdown, no code fences.
    Use exact table and column names from the schema.
    If any database, schema, table, or column name contains special characters (such as hyphens, spaces, or reserved keywords), wrap it in backticks (`). 
    For Example: 
    Correct: SELECT name FROM my-db`.`user table`; 
    Incorrect: SELECT name FROM my-db.user table;
    Ensure the SQL syntax is valid for MySQL.
    Avoid any newline characters or formatting that could cause SQL syntax errors — output as a single line unless explicitly needed.
    Never include comments or extra text — just the SQL query.

    For example:
    Question: Which 3 members have the most users?
    SQL Query: select m.name as member_name, count(u.id) as user_count from `wnxa-staging`.Member m JOIN `wnxa-staging`.User u on u.memberId = m.id GROUP BY m.id ORDER BY user_count DESC LIMIT 3;
    Question: How many active members are there?
    SQL Query: select count(*) from `wnxa-staging`.Member where isActive = 1;

    Replace wnxa-staging with the actual database name if needed.

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOllama(model=sql_chain_model, temperature=llm_temperature)

    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text, db_name=lambda _: db_name)
        | prompt
        | llm
        | StrOutputParser()
    )

# Get Response
def get_response(user_query: str, db: SQLDatabase, chat_history: list, schema_text: str, db_name: str) -> str:
    
    # Build SQL Chain
    sql_chain = get_sql_chain(schema_text, db_name)

    template = """
    You are a Data Analyst for the WNXA MySQL database. You are interacting with a user who is asking you questions about the database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    Execute the SQL query and return the result in human language.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=sql_response_model, temperature=llm_temperature)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: schema_text, 
            response=lambda vars: db.run(clean_sql(vars["query"])),
            )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the chain with the user query and chat history
    response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
    return response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.chat_history.append(AIMessage(content="Hello! Ask me anything about the WNXA MySQL database."))

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
                    st.session_state["db_host"],
                    st.session_state["db_user"],
                    st.session_state["db_password"],
                    st.session_state["db_name"],
                    st.session_state["db_port"]
                )

                st.session_state.schema_text = generate_schema_description(
                    st.session_state["db_host"],
                    st.session_state["db_user"],
                    st.session_state["db_password"],
                    st.session_state["db_name"],
                    st.session_state["db_port"]
                )

                st.success("Connected to the database!")
            except Exception as e:
                st.error(f"Failed to connect: {e}")
            
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.chat_message("AI").markdown(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("Human").markdown(message.content)

user_query = st.chat_input("Type your message here...")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    logging.info(f"User query: {user_query}")

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            # If database is not initialized, show an error
            if "db" not in st.session_state:
                response = "Please connect to the database first."
            else:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history, st.session_state.schema_text, st.session_state.db_name)
                logging.info(f"Response generated")
            st.markdown(response)
        except Exception as e:
            st.error(f"Error processing your request: {e}")
            response = "Sorry, I couldn't process your request. Please rephrase your question."

        
    st.session_state.chat_history.append(AIMessage(content=response))