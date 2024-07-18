from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,FewShotChatMessagePromptTemplate,PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

examples = [
    {
        "input": "List all the cities with population greater than 10,00,000.",
        "query": "SELECT name FROM city WHERE population > 1000000;"
    },
    {
        "input": "Get the highest population in a country.",
        "query": "SELECT MAX(population) FROM country;"
    },
    {
        "input": "Get the country with highest population.",
        "query": "SELECT name FROM country WHERE country.population IN (SELECT MAX(population) FROM country);"
    },
    {
        "input": "Name the database.",
        "query": "SELECT DATABASE();"
    },
    {
        "input": "What are the contents in the DB?",
        "query": "SHOW TABLES;"
    }
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

vectorstore = Chroma()
vectorstore.delete_collection()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    vectorstore,
    k=2,
    input_keys=["input"],
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input","top_k"]
)


final_prompt = ChatPromptTemplate.from_messages(
     [
         ("system", "You are a MySQL expert named SQLChatbot. Given an input question, create a syntactically correct MySQL query to run. Unless otherwise specificed.\n\nHere is the relevant table info: {table_info}\n\nBelow are a number of examples of questions and their corresponding SQL queries. Those examples are just for referecne and should be considered while answering follow up questions and if questions are general and not related to sql try to answer them without giving error message."),
         few_shot_prompt,
         MessagesPlaceholder(variable_name="messages"),
         ("human", "{input}"),
     ]
)

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)