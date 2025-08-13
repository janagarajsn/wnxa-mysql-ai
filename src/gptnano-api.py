import os
from dotenv import load_dotenv
import streamlit as st
import logging
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

st.set_page_config(page_title="WNXA MySQL AI", page_icon=":speech_balloon:", layout="wide")
st.title("Chat with WNXA MySQL DB")

# Initialize the MySQL connection
def init_database(host: str, user: str, password: str, database: str, port: str) -> SQLDatabase:
    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Get SQL Chain
def get_sql_chain(db: SQLDatabase):
    prompt_template = """
    You are a Data Analyst for the WNXA MySQL database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>
    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: Which 3 members have the most users?
    SQL Query: select m.name as member_name, count(u.id) as user_count from wnxa-staging.Member m JOIN wnxa-staging.User u on u.memberId = m.id GROUP BY m.id ORDER BY user_count DESC LIMIT 3;
    Question: How many active members are there?
    SQL Query: select count(*) from wnxa-staging.Member where isActive = 1;

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Get Response
def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    
    # Build SQL Chain
    sql_chain = get_sql_chain(db)

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

    llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.0)

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(), 
            response=lambda vars: db.run(vars["query"]),
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
                response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error processing your request: {e}")
            response = "Sorry, I couldn't process your request. Please rephrase your question."

        
    st.session_state.chat_history.append(AIMessage(content=response))