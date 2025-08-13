import os
from dotenv import load_dotenv
import streamlit as st
import pymysql
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_model = os.getenv("LLM_MODEL", "deepseek-r1:14b")
llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))

st.set_page_config(page_title="WNXA MySQL AI", page_icon=":speech_balloon:", layout="wide")
st.title("Chat with WNXA MySQL DB")

# Generate Schema Description
def generate_schema_description(host, user, password, database, port=3306) -> str:
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
            cursor.execute(f"SHOW TABLES")
            tables = [row[f'Tables_in_{database}'] for row in cursor.fetchall()]
            
            for table in tables:
                schema_description.append(f"Table: {table}")
                cursor.execute(f"SHOW COLUMNS FROM `{table}`")
                columns = cursor.fetchall()
                
                for col in columns:
                    col_name = col['Field']
                    col_type = col['Type']
                    nullable = "NULL" if col['Null'] == 'YES' else "NOT NULL"
                    schema_description.append(f"- {col_name} ({col_type}, {nullable})")
                
                schema_description.append("")  # blank line between tables
    
    finally:
        connection.close()
    
    return "\n".join(schema_description)

# Log and run the query
def log_and_run_query(db: SQLDatabase, query: str):
    print(f"Executing SQL Query: {query}")
    return db.run(query)

# Initialize the MySQL connection
def init_database(host: str, user: str, password: str, database: str, port: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Get SQL Chain
def get_sql_chain(db: SQLDatabase, schema_text: str):
    prompt_template = """
    You are a Data Analyst for the WNXA MySQL database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}

    
    Important instructions:
    - Write only the pure SQL query without any markdown, code fences, or extra explanation.
    - Use the exact table and column names as provided in the schema.
    - Always wrap database, schema, table, or column names with backticks (`) if they contain special characters like hyphens.
    - Do NOT include any comments or formatting besides the SQL code itself.
    - Output only the SQL query, no explanations, no markdown, no code fences.
    - Do NOT include words like 'sql', 'SQL Query:', or any prefixes before the query.
    - Use exact table/column names from the schema.
    - Wrap table or column names in backticks (`) if they contain special characters.
    - Ensure the SQL syntax is correct for MySQL.
    - Avoid nextline character or any character that causes sql syntax errors.

    For example:
    Question: Which 3 members have the most users?
    SQL Query: select m.name as member_name, count(u.id) as user_count from `wnxa-staging`.Member m JOIN `wnxa-staging`.User u on u.memberId = m.id GROUP BY m.id ORDER BY user_count DESC LIMIT 3;
    Question: How many active members are there?
    SQL Query: select count(*) from `wnxa-staging`.Member where isActive = 1;

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOllama(model=llm_model, temperature=llm_temperature)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=lambda _: schema_text)
        | prompt
        | llm
        | StrOutputParser()
    )

# Get Response
def get_response(user_query: str, db: SQLDatabase, chat_history: list, schema_text: str):
    
    # Build SQL Chain
    sql_chain = get_sql_chain(db, schema_text)

    template = """
    You are a Data Analyst for the WNXA MySQL database. You are interacting with a user who is asking you questions about the database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(model=llm_model, temperature=llm_temperature)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: schema_text, 
            response=lambda vars: log_and_run_query(db, vars["query"]),
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
    st.session_state.chat_history = [
        AIMessage(content="Hello! Ask me anything about the WNXA MySQL database."),
    ]

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
                db = init_database(
                    st.session_state["db_host"],
                    st.session_state["db_user"],
                    st.session_state["db_password"],
                    st.session_state["db_name"],
                    st.session_state["db_port"]
                )
                st.session_state.db = db

                # Generate and cache schema text
                schema_text = generate_schema_description(
                    host=st.session_state["db_host"],
                    user=st.session_state["db_user"],
                    password=st.session_state["db_password"],
                    database=st.session_state["db_name"],
                    port=st.session_state["db_port"]
                )
                st.session_state.schema_text = schema_text

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

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        try:
            # If database is not initialized, show an error
            if "db" not in st.session_state:
                response = "Please connect to the database first."
            elif "schema_text" not in st.session_state:
                response = "Schema information is not available. Please reconnect to the database."
            else:
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history, st.session_state.schema_text)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error processing your request: {e}")
            response = "Sorry, I couldn't process your request. Please rephrase your question."

        
    st.session_state.chat_history.append(AIMessage(content=response))