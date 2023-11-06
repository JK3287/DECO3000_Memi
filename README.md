# DECO3000 A5 MEMI - by Veronikah McClelland and Justin Kwon

## Introduction
Memi is an app that employs artificial intelligence, as a solution made in order to tackle mental issues that come with age.

The first page, **TALK** is in the form of a Large Language Model chatbot that uses the ChatGPT-3.5 model in order to provide information. Firstly, the user should put their input in the form of text. A Large Language Model (LLM) should take in this input and should formulate a response through the perspective of a reminiscent therapist, with an audible response available. Through Neuro-Linguistic Programming, the conversation is scanned to find potential keywords, and finally through LangChain it ends up summarised. When the user asks for a friend to find, using the Spacy keywords, the chatbot will go through a database of people and try to match keywords in order to find compatitable people.

The second page, **FRIENDS** is built to host a list of friends that the user has found through a consultation with the chatbot from the TALK page.

## Setup
In order to setup Memi, it is recommended that you have the following: an **Integrated Development Environment (IDE)**, **Conda**, an **OpenAI Account** and an **OpenAI API Key**. Once you have the aforementioned components, ensure that you are running the Python code through a Conda terminal. Afterwards, open the **Memi** folder on your IDE and please make sure these are installed:

### Streamlit - Simplistic Interface
* streamlit
* streamlit_option_menu
* streamlit_chat
* audio_recorder_streamlit

### OpenAI - Large Language Models
* openai
* langchain
* gtts

### .env - Hidden Files
* os
* dotenv

### Spacy - Natural Language Processing
* spacy
* pandas

## Final Steps
In order to get the program running, please make the following hidden files/folders

* .env (file) - just write one line of code that goes like this: "OPENAI_API_KEY ='your_key_here' "
* .gitignore (file) - add the .env file on the top of the file

* .streamlit (folder) 
- config.toml (file) - customise it at your own will:

Once you get this done, on a Conda terminal, type in streamlit run 'pathname/memi_main.py', and Memi should open up as a new tab on your browser.


## How to use Memi
Memi will start on all cases, at the **TALK** page. There you have two options to provide input for the chatbot: to type in using **text**, or to record your own **voice**. Once you provide your input, wait a few seconds and the chatbot will provide you with a response based on reminiscence therapy. Once you have talked enough to the chatbot, ask it to find friends. As seen on the bottom, the chatbot generates keywords and a summary based on your conversation. Soon it will find friends using keywords. Then go to the **FRIENDS** page.