# A simple introduction to Vector Search
The demo is created to illustrate the basics of vector search using third-party APIs and open source models that could be run locally on CPU.
You're free to play around with different models for embeddings and modify the code to run on GPU as well.

## Setup
Create a virtual environment and activate it.
```python
python -m venv <env_name>
source <env_name>/bin/activate
```

## Clone the repository
Clone the repository.
```python
git clone https://github.com/AbinayaM02/DevFest2023_VectorSearch.git
cd DevFest2023_VectorSearch
```

## Install requirements
Install the necessary dependencies by executing the following command.
```python
pip install -r requirements.txt
```

## Run the demo
Once the setup is completed, download the embeddings file created for the demo variation 2 as per the instructions mentioned [here](https://github.com/AbinayaM02/DevFest2023_VectorSearch/blob/main/data/README.md) and run the following command.
```python
cd scripts
streamlit run search_demo.py
```

## Cloud demo
The demo is also deployd to Streamlit cloud. You can run it directly from [here](https://devfest2023vectorsearch.streamlit.app/).
It is higly recommended to run the demo locally for demo variation 2. 

