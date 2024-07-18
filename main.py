import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ChatMessageHistory
from prompts import final_prompt, answer_prompt
from table_details import select_table

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

db_user="root"
db_pass="rahulbs"
db_host="localhost"
db_port="3306"
db_name="world"
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

llm=ChatOpenAI(openai_api_key=api_key,temperature=0)

generate_query = create_sql_query_chain(llm, db,final_prompt)
execute_query = QuerySQLDataBaseTool(db=db)
rephrase_answer = answer_prompt | llm | StrOutputParser()

chain = (
RunnablePassthrough.assign(table_names_to_use=select_table) |
RunnablePassthrough.assign(query=generate_query).assign(
    result=itemgetter("query") | execute_query
)| rephrase_answer
)

history = ChatMessageHistory()

if __name__ == "__main__":
    while True:
        try:
            question=input("\nYou: ")
            response = chain.invoke({"question": question,"messages":history.messages})
            history.add_user_message(question)
            history.add_ai_message(response)
            print("\nSQLChatbot:",response)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("\nError occurred the bot was not able to satisfy your request",e)
