import streamlit as st
import os
import json
from scipy import spatial
from datetime import date, timedelta, datetime
from newsapi import NewsApiClient
from newspaper import Article
from newspaper import Config

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()
newsapi = NewsApiClient(api_key=os.getenv('news_api_key'))

# Set up JSON Formatting
class QuerySchema(BaseModel):
    queries: list[str] = Field(description="the list of TOP 10 queries")

parser = JsonOutputParser(pydantic_object=QuerySchema)
format_instructions = parser.get_format_instructions()

# Date variables
today = datetime.today()
prev_30day = today - timedelta(days=30)
today = today.strftime('%Y-%m-%d')
prev_30day = prev_30day.strftime('%Y-%m-%d')

# Helper Functions
def cosine_similarity(x,y):
    return 1 - spatial.distance.cosine(x, y)

def get_embeddings(text):
    embeddings = OpenAIEmbeddings(model=EMBEDDING)
    response = embeddings.embed_query(text)
    return response

# Get Response from News API
def search_news(query, num_articles=5, from_datetime = prev_30day,to_datetime = today):
    
    response = newsapi.get_everything(q=query,
                                      from_param=prev_30day,
                                      to=today,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=num_articles
                                      )


    return response

# Get Content From URL
user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent

def get_content_from_url(url, text_max_length=2000):
    try:
        page = Article(url)
        page.download()
        page.parse()
        
        return page.text[:text_max_length]
    
    except:
        return 'Error'


# Main Functions
def get_response(user_query):

    template = """
                You will be generating search queries to find recent news articles on a given topic using a NEWS API.\

                The topic is: {user_query}\

                Your goal is to generate many potential search queries that are relevant to the topic. To do this:\
                - Use different keywords and phrases related to the topic \
                - Vary the specificity of your queries, making some more narrow and others more broad\
                - Be creative and come up with as many distinct query ideas as you can\

                First, brainstorm at least 20 distinct query ideas of varying specificity. Please don't return the output of brainstorming.\

                Then select the 10 best, most promising queries. Provide them in the following format: {format_instructions}
                
                """

    prompt = PromptTemplate.from_template(template,partial_variables={"format_instructions": parser.get_format_instructions()})
    
    llm = ChatOpenAI(model=MODEL)
        
    chain = prompt | llm 
    
    return chain.stream({
        "user_query": user_query,
    })

def generate_hypothetical_answer(user_query):
    template = """
                Here is a question from a user:{user_query}

                Please make up a hypothetical answer to this question. Imagine you have all the details needed to answer it, even if those details are not real. 
                Do not use actual facts in your answer. Instead, use placeholders like 'EVENT affected something' or 'NAME mentioned something on DATE' to represent key details.
                Limit your answer in 500 characters.

                
         """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model=MODEL)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "user_query": user_query,
    })

def pick_news(queries,hypothetical_answer_embedding):
    articles = []
    for query in queries:
        result = search_news(query)
        if result['status'] == 'ok':
            articles = articles + result['articles']
        else:
            raise Exception(result["message"])
            
    #Remove duplicates
    articles = {article["url"]: article for article in articles}
    
    # Get Content From url
    for key in articles.keys():
        content_from_url = get_content_from_url(key)
        if content_from_url!='Error':
            articles[key]['content'] = content_from_url
            
    articles = list(articles.values())
    
    
    articles_prepare_embedd =  [
        f"{article['title']} {article['content'][0:500]}"
        for article in articles
    ]
    
    article_embeddings =  [get_embeddings(article) for article in articles_prepare_embedd]
    
    cosine_similarities = []
    for article_embedding in article_embeddings:
        cosine_similarities.append(cosine_similarity(hypothetical_answer_embedding, article_embedding))
        
    scored_articles = zip(articles, cosine_similarities)
    sorted_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)


    formatted_top5_results = [
        {
            "Title": article["title"],
            "Url": article["url"],
            "Content": article['content']
        }
        for article, _score in sorted_articles[0:5]
    ]
    
    return formatted_top5_results

def summarize_top5_reuslts(formatted_top5_results,user_query):
    template = """
                Generate an answer to the user's question based on the given search results.
                TOP_RESULTS: {formatted_top5_results}
                USER_QUESTION: {user_query}

                Include as many details as possible in the answer. 
                Reference the relevant search result urls as markdown links.
                

                
         """

    prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model=MODEL)

    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "user_query": user_query,
        "formatted_top5_results": formatted_top5_results
    })



# Set Streamlit page configuration
st.set_page_config(page_title="🗞️News Feed Assistant🤖", layout="centered")

## Hide header
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("🗞️News Feed Assistant🤖")
st.markdown(
    """ 
        > :black[**A Chatbot for News,** *powered by -  [LangChain](https://python.langchain.com/v0.2/docs/introduction/) + 
        [OpenAI](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4) + 
        [Streamlit](https://streamlit.io) + 💪 **Audit DS Team***]
        """
)

# Set up sidebar with various options
with st.sidebar.expander(" 🛠️ Settings ", expanded=False):

    MODEL = st.selectbox(
        label="LLM_Model",
        options=[
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
    )
    
    EMBEDDING = st.selectbox(
        label="Embedding_Model",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
    )
    
# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your assistant, here to help you find news that interests you. How may I assist you today?"),
    ]

    
# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI",avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message('Human', avatar="🧐"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message('Human', avatar="🧐"):
        st.markdown(user_query)

    # Generate Related Queries
    st.markdown("**Relevant Queries:**")
    with st.chat_message("AI",avatar="🤖"):
        response = st.write_stream(get_response(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))
    queries = json.loads(response)['queries']
    queries.insert(0,user_query)

    # Generate Hypothetical Answer
    st.markdown("**Hypo Answers:**")
    with st.chat_message("AI", avatar="🤖"):
        response = st.write_stream(generate_hypothetical_answer(user_query))

    st.session_state.chat_history.append(AIMessage(content=response))
    hypothetical_answer_embedding = get_embeddings(response)


    # Retrieving TOP5 Articles
    with st.spinner("Retrieving Articles..."):
        formatted_top5_results = pick_news(queries,hypothetical_answer_embedding)
        formatted_top5_results_display = []
        for result in formatted_top5_results:
                formatted_top5_results_display.append('\n\n'.join([":".join([f'**{key}**',value]) for key,value in result.items()]))
        formatted_top5_results_display = f"\n\n{'-'*50}\n\n".join(formatted_top5_results_display)

    st.markdown("**TOP 5 Articles:**")
    with st.chat_message("AI", avatar="🤖"):
        st.write(formatted_top5_results_display)

    # Summarized Output
    st.markdown("**Summarized Output:**")
    with st.chat_message("AI", avatar="🤖"):
        response = st.write_stream(summarize_top5_reuslts(formatted_top5_results,user_query))

    st.session_state.chat_history.append(AIMessage(content=response))