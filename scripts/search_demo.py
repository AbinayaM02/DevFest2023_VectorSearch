import streamlit as st
from wiki_embed import load_data_complete, get_embeddings, get_top_k
from config import MAX_DOCS, WIKI_DATA, EMB_MODEL, TOP_K, WIKI_EMB, WIKI_EMB_MODEL
from wiki_cohere import load_data, get_query_embeddings, get_top_k_cohere
from tempfile import NamedTemporaryFile
from own_data_search import get_document, get_embeddings_model, index_embeddings, get_response

def intro():
    st.write("# Welcome to Vector Search Demo! 👋")
    st.sidebar.success("Select a demo variation.")

    st.markdown(
        """
        ### Wikipedia at your Fingertips
        Wikipedia is one of the easily accessible source of information.
        What if we could run Wikipedia search locally in our computer.
        The first two demos are aimed at showing how Large Language Models (LLM) 
        can be used to perform vector search on Wikipedia data. 
        
        #### Data:
        The data is sourced from Hugging Face,
        - [Cohere/wikipedia-22-12](https://huggingface.co/datasets/Cohere/wikipedia-22-12) - Contains the dump of 
          Wikipedia articles without embeddings
        - [Cohere/wikipedia-22-12-en-embeddings](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings) - 
          Contains the dump of Wikipedia articles with embedding
        - Your own pdf (imagine a book)

        #### Model:
        The following are the models used to get the embeddings of the data,
        - multilingual-22-12 (Cohere model)
        - [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

        #### Demo Options:
        **All these demos are meant to be run on CPU and for English langauge data.** 

        ##### 1. Search using Cohere API
        Cohere released the Wikipedia Embeddings data where Wikipedia articles
        are split into paragraphs and theri embeddings are obtained using their
        propreitry model multilingual-22-12. This demo requires Cohere API key
        to get embeddings of the query. Only specified amount of documents are
        streamed without ahving to download the complete 35 million datapoints.

        ##### 2. Search Offline
        In this demo, the complete Wikipedia data (Cohere/wikipedia-22-12) which
        has articles split into paragraphs is locally downloaded and their embeddings
        are generated using BAAI/bge-large-en-v1.5 model for a specified number of 
        documents (MAX_DOCS). The embeddings are then indexed for future use so that 
        we don't have to generate embddings every time we run the demo. Search is 
        performed on the indexed embeddings and the corresponding documents are
        retrieved.
        
        ### Search Your Own Data
        What if you could perform search on your local documents. This demo shows how
        you can upload a pdf and search for answers within the pdf (image a pdf book).
        
        #### App:
        This demo is built using streamlit which is an open-source app framework 
        built specifically for Machine Learning and Data Science projects. 

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what kind of search demo you want to look at!

        #### Libraries used:
        - [Cohere](https://cohere.com/)
        - [LangChain](https://www.langchain.com/)
        - [FAISS](https://github.com/facebookresearch/faiss)
        - [Sentence Transformers](https://www.sbert.net/)

        #### Github Link:
        - [Vector Search Demo]()

        #### Want to learn more about streamlit?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
    """
    )

def search_offline():
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    st.write(
        """
        This demo illustrates how Wikipedia data can be searched offline using
        open source model. When you run this demo for the first time, both data loading
        (35 million datapoints) and generation of embeddings takes a lot of time 
        proportional to the number of pages being loaded. After the first run, the 
        embeddings file will be saved and loaded in the subsequent run. If you change 
        the number of pages everytime you run the demo, please delete the old embeddings 
        file and the run the demo. Also the result will be dependent on the chosen model 
        and the number of documents loaded. Feel free to play around with differnet models
        from [here](https://huggingface.co/spaces/mteb/leaderboard) and data subsets 
        (check the config file for changing the settings). Enjoy!
        """
    )

    st.info("For the purpose of the demo, choose 100000 on the slider "+ 
            "since the embeddings are already generated!")
    pages = st.slider("Select no. of pages to load:", 
                              min_value=100, 
                              max_value=MAX_DOCS, 
                              step=100,
                              value=1000)

    # Get the query from the user
    query = st.text_input("What do you want to search?")

    if query != '':
        with st.status("Loading data..."):
            # Load data
            docs = load_data_complete(max_docs=pages, 
                            wiki_emb=WIKI_DATA)
        st.info("Data loaded")

        with st.status("Getting document embedding..."):
            # Get docs embeddings
            doc_embeddings = get_embeddings(emb_model=EMB_MODEL,
                                            docs=docs, 
                                            query_str=False)
        st.info("Document embeddings is loaded")

        with st.status("Getting query embedding..."):
            # Get query embeddings
            query_embedding = get_embeddings(docs=[query], 
                                            emb_model=EMB_MODEL,
                                            query_str=True)
        st.info("Query embedding is obtained")

        with st.status("Searching..."):
            # Get top k documents matching the query
            titles, texts = get_top_k(query, docs, query_embedding, 
                                      doc_embeddings, TOP_K)

        st.write("Search Results: \n")
        for i in range(len(titles)):
            st.write(titles[i] + ":")
            st.write(texts[i] + "\n")


def search_cohere():
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        This demo illustrates how Wikipedia data can be searched using Cohere's
        embedding model. Cohere API key is needed for this demo. Please obtain the API
        key from [here](https://dashboard.cohere.com/api-keys). The data is streamed in 
        this demo to reduce memory overload. Cohere model that is used to create the 
        Wikipedia embedding is used for obtaining query embedding too. Enjoy!
        """
    )

    pages = st.slider("Select no. of pages to load:", 
                              min_value=100, 
                              max_value=MAX_DOCS, 
                              step=100,
                              value=1000)

    # Get the query from the user
    query = st.text_input("What do you want to search?")

    if query != '':
        # Get the Cohere API key
        cohere_token = st.text_input('Cohere API Key', 
                                 type='password', 
                                 disabled=not query)

        if cohere_token != '':
            with st.status("Loading data and document embeddings..."):
                # Load data
                docs, doc_embeddings = load_data(max_docs=pages, 
                                                wiki_emb=WIKI_EMB)
            st.info("Data and document embeddings loaded")

            with st.status("Getting query embedding..."):
                # Get query embeddings
                query_embedding = get_query_embeddings(query=[query], 
                                                    model=WIKI_EMB_MODEL,
                                                    token=cohere_token)
            st.info("Query embedding is obtained")

            with st.status("Searching..."):
                # Get top k documents matching the query
                titles, texts = get_top_k_cohere(query, docs, query_embedding, 
                                        doc_embeddings, TOP_K)

            st.write("Search Results: \n")
            for i in range(len(titles)):
                st.write(titles[i] + ":")
                st.write(texts[i] + "\n")

def search_own_data():
    st.markdown(f'# {list(page_names_to_funcs.keys())[3]}')
    st.write(
        """
        This demo illustrates how you can search your own document using
        LangChain. You can upload a pdf of a textbook and search for topics
        within the book. It can be a handy tutor. Enjoy!
        """
    )

    # File upload
    uploaded_file = st.file_uploader('Upload an article:', type='pdf')

    # Query text
    query_text = st.text_input('Enter your question:', 
                            placeholder = 'Please provide a short summary.',
                            disabled=not uploaded_file)
    
    result = []
    with st.form('myform', clear_on_submit=True):
        submitted = st.form_submit_button('Submit', 
                                          disabled=not(uploaded_file and 
                                                       query_text))
        if submitted:
            with st.spinner('Searching the document...'):
                # Create temp file once the pdf is uploaded
                with NamedTemporaryFile(dir='.', suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    documents = get_document(tmp_file.name)
                # Create faiss index with embeddings
                faiss = index_embeddings(documents, embeddings=get_embeddings_model(EMB_MODEL))
                # Get response
                response = get_response(faiss, query_text)

                if len(response):
                    st.write("Search Results: \n")
                    for i in range(len(response)):
                        st.write("Page Number: " + str(response[i].metadata['page']+1))
                        st.info("Page Content: " + response[i].page_content)
                        
                

page_names_to_funcs = {
    "Home": intro,
    "Search with Cohere API": search_cohere,
    "Search Offline": search_offline,
    "Search your own data": search_own_data
}


# Create sidebar
st.sidebar.title("Demo Options!!!")
demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()