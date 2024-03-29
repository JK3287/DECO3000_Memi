# DECO3000 A5 MEMI - by Veronikah McClelland and Justin Kwon

## Introduction
Memi is an app that employs artificial intelligence, as a solution made in order to tackle mental issues that come with age.

The first page, **TALK** introduces Memi as a chatbot that uses the ChatGPT-3.5 model in order to provide information. Firstly, the user should put their input in the form of text. Memi should take in this input and should formulate a response through the perspective of a reminiscent therapist, with an audible response available. Through Neuro-Linguistic Programming, the conversation is scanned to find potential keywords. When the user asks for a friend to find, using the Spacy keywords, Memi will go through a database of people and try to match keywords in order to find compatitable people. These people can be added to the **FRIENDS** page.

The second page, **FRIENDS** is built to host a list of friends that the user has found through a consultation with Memi from the TALK page.

The third page, **CHATLOG** is where conversations with the Memi are stored. In addition, summaries generated by LangChain are there.


## Setup
To avoid compatibility issues,as the majority of program was made on Justin's computer, macOS (v12.0.1), in order to setup Memi, it is recommended that you have the following: **Virtual Studio (v1.76.1)** with **Python (v3.11)** and installed **Conda (v23.9.0)**, an **OpenAI Account** and an **OpenAI API Key**. Once you have the aforementioned components, ensure that you are running the Python code through a Conda terminal. Afterwards, open the **Memi** folder on your IDE and please make sure these are installed:

### Streamlit - Simplistic Interface
* streamlit (v1.27.2)
* streamlit_option_menu (v0.3.6)
* streamlit_chat (v0.1.1)
* audio_recorder_streamlit (v0.0.8)

### OpenAI - Large Language Models
* io (Pre-installed)
* openai (v0.28.1)
* langchain.llms (v0.0.329)
* gTTS (v2.4.0)

### .env - Hidden Files
* os (Pre-installed)
* dotenv (v1.0.0)

### Spacy - Natural Language Processing
* spacy (v3.7.2)
* pandas (v1.5.3)

## Final Steps
In order to get the program running, please make the following hidden files/folders

* .env (file) - just write one line of code that goes like this: "OPENAI_API_KEY ='your_key_here' "
* .gitignore (file) - add the .env file on the top of the file

* .streamlit (folder) 
- config.toml (file) - customise it at your own will:

Once you get this done, on a Conda terminal, type in streamlit run 'pathname/memi_main.py', and Memi should open up as a new tab on your browser.


## How to use Memi
Memi will start on all cases, at the **TALK** page. Please follow all instructions on the page.

There you have two options to provide input for the chatbot: to type in using **text**, or to record your own **Voice**. There is a third option **stop**, which is used to prevent unintentional sending of text messages. 

Memi runs on this prompt template: *You are a chatbot named Memi. I want you to remind me, an elderly human, to look at the bottom of the page to find a list of potential friends once I talk enough about myself when I want to make a friend, otherwise I want you to assist me with creating a conversation that looks into past positive memories to put me in a better mood.  I want you to inquire about details of the memories I am telling you about, so we can get a better understanding of what happened.  Start with asking me about what I want to talk about, but if I am not sure, please prompt me with a conversation starter to help.  Please keep the conversation in a positive light, and help me to appreciate my past experiences. Do not make answers longer than 50 words. Your tone is friendly and casual, yet caring and empathetic.*

Once you provide your input, wait a few seconds and the chatbot will provide you with a response based on reminiscence therapy.

If you want to save a conversation between you and Memi, click on the button underneath the **Save Chat** subheader. After that, go to the **CHATLOG** page to see if Memi stores your conversations and check for a summary.

As seen on the bottom, the chatbot generates keywords. Soon it will find friends using keywords. In order to add friends, click on the buttons below each name. Then go to the **FRIENDS** page.


