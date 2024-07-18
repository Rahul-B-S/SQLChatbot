import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain.memory import ChatMessageHistory

def get_table_details():
    # Read the CSV file into a DataFrame
    table_description = pd.read_csv("table_details.csv")
    table_docs = []

    # Iterate over the DataFrame rows to create Document objects
    table_details = ""
    for index, row in table_description.iterrows():
        table_details = table_details + "Table Name:" + row['Table'] + "\n" + "Table Description:" + row['Description'] + "\n\n"

    return table_details

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

db_user="root"
db_pass="rahulbs"
db_host="localhost"
db_port="3306"
db_name="world"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

llm=ChatOpenAI(openai_api_key=api_key,temperature=0)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

rephrase_answer = answer_prompt | llm | StrOutputParser()

examples = [
    {
        "input": "List all customers in France with a credit limit over 20,000.",
        "query": "SELECT * FROM customers WHERE country = 'France' AND creditLimit > 20000;"
    },
    {
        "input": "Get the highest payment amount made by any customer.",
        "query": "SELECT MAX(amount) FROM payments;"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    input_variables=["input","top_k"]
)

final_prompt = ChatPromptTemplate.from_messages(
     [
         ("system", "You are a MySQL expert. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries. Those examples are just for referecne and hsould be considered while answering follow up questions"),
         few_shot_prompt,
         MessagesPlaceholder(variable_name="messages"),
         ("human", "{input}"),
     ]
)

generate_query = create_sql_query_chain(llm, db,final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)

class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")

# table_names = "\n".join(db.get_usable_table_names())
table_details = get_table_details()

table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt)
def get_tables(tables: List[Table]) -> List[str]:
    tables  = [table.name for table in tables]
    return tables
select_table = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables

chain = (
RunnablePassthrough.assign(table_names_to_use=select_table) |
RunnablePassthrough.assign(query=generate_query).assign(
    result=itemgetter("query") | execute_query
)
| rephrase_answer
)

history = ChatMessageHistory()
while True:
    question=input("You:")
    if question.lower()=="q":
        break
    response = chain.invoke({"question": question,"messages":history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    print("Echo:",response)
