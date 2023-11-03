# streamlit run '/Users/yungvenuz/Documents/Uni/Year 3 DC/DECO3000/DECO3000_Memi/a.py'

# imports

import openai
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_chat import message
from audio_recorder_streamlit import audio_recorder

from gtts import gTTS

from io import BytesIO
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pandas as pd

import spacy
nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words |= {"hey","uh","ah","oh","aw", "sorry", "hear", "feeling", "way", "reflecting", "positive", "memories", "help", "lift",
                            "spirits", "particular", "memory", "experience", "comes", "mind", "think", "happy", "time", "let", "prompt",
                            "tell", "time", "felt", "truly", "alive", "joyful", "assist", "support", "sounds", "like", "use", "boost", "positivity", 
                            "like", "talk", "favorite", "sure", "start", "ask","question", "pleasant", "exciting", "taken", "sounds", "amazing", 
                            "special", "memorable", "specific", "moment", "stood", "looking", "mood", "remind", "happier", "times", "suggest", 
                            "conversation", "starter", "going", "provide", "need", "important", "reach", "mental", "health", "professional", 
                            "trusted", "person", "life", "understand", "feel", "break", "focus", "moments", "brings", 
                            "joy", "makes", "feel", "totally", "normal", "days", "bit", "disconnected", "topic", "fun",
                            "achievement", "proud", "situation", "thing", "things", "completely", "certain", "bring", "wave", "nostalgia",
                            "natural", "ups", "downs", "kind", "enjoy", "left", "right", "miss", "missed", "reminiscence", "reminisce", "reminisced", "reminisces", 
                            "reminiscent", "maybe"}
import spacy_streamlit

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Chatbot",
    page_icon=":robot:"
)

# CSS SOURCE
st.markdown(
    """
<style>
* {
   font-family: Montserrat;
   text-align: center;
}

.stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
        transform: scale(1.25);
    }

</style>
""",
    unsafe_allow_html=True,
)

# MENU

potential_friends = []

with st.sidebar:
    selected = option_menu(
        menu_title="Memi",  # required
        options=["TALK", "FRIENDS"],  # required
    )

# TALK CHATBOT
if selected == "TALK":
    
    # Header
    st.title("TALK")
    st.header(" ")
    st.header("Feel free to talk to an AI chatbot, who will provide you with reminiscence therapy.")
    st.header(" ")
    unique_keywords = set()

    # Storing GPT-3.5 responses for easy retrieval to show on Chatbot UI in Streamlit session
    if 'prompted' not in st.session_state:
        st.session_state['prompted'] = []

    # Storing user responses for easy retrieval to show on Chatbot UI in Streamlit session
    if 'stored' not in st.session_state:
        st.session_state['stored'] = []

    # Storing entire conversation in the required format of GPT-3.5 in Streamlit session
    if 'full_conversation' not in st.session_state:
        st.session_state['full_conversation'] = [{'role': 'system',
                                                  'content': 'You are Memi. I want you to assist me with creating a conversation that looks into past positive memories to put me in a better mood.  I want you to inquire about details of the memories I am telling you about, so we can get a better understanding of what happened.  Start with asking me about what I want to talk about, but if I am not sure, please prompt me with a conversation starter to help.  Please keep the conversation in a positive light, and help me to appreciate my past experiences.  Do not make answers longer than 50 words. Do not use interjections. Your tone is friendly and casual, yet caring and empathetic.'}]

    if 'conversation_keywords' not in st.session_state:
        st.session_state['conversation_keywords'] = []

    # Text to speech function
    def text_to_speech(text):
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()

    # Response function
    def query(user_text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.full_conversation
        )

        message = response.choices[0].message.content
        # Adding bot response to the
        st.session_state.full_conversation.append({'role': 'system', 'content': message})
        bot_keywords = extract_keywords(message)
        st.session_state.conversation_keywords.extend(bot_keywords)
        return message

    # Extract (NLP)
    def extract_keywords(text):
        doc = nlp(text)
        keywords = [token.text for token in doc if not token.is_stop and token.is_alpha]
        unique_keywords.update(keywords)
        return keywords

    # Summarize
    def summarize_text(chat_log):
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        docs = [Document(page_content=t) for t in chat_log]
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        summary = chain.run(docs)
        return summary

    # Display
    def display_chat_summary(chat_log):
        st.header(" ")
        st.markdown("#### Chat Summary", unsafe_allow_html=True)
        summary = summarize_text(chat_log)
        st.write(summary)

    # Find friends
    def find_friend(unique_keywords):
        data = pd.read_excel('database.xlsx')

        potential_friends = []

        for index, row in data.iterrows():
            keywords = row['KEYWORD'].split(', ')
            matching_keywords = set(unique_keywords) & set(keywords)

            if len(matching_keywords) > 0:
                potential_friends.append({
                    'Name': row['NAME'],
                    'Matching Keywords': ', '.join(matching_keywords)
                })
        
        return potential_friends

    # Initialize the list
    all_user_messages = []

    # Input function
    def get_text():
        st.markdown("#### Choose input method:", unsafe_allow_html=True)
        input_option = st.radio("", ("Text", "Record Voice"))

        if input_option == "Text":
            st.header(" ")
            st.markdown("#### Please type in the box below.", unsafe_allow_html=True)
            input_text = st.text_input("", " ", key="input")
        else:
            # Record audio using audio_recorder ONLY WORKS IF TEXT IS FIRST
            st.header(" ")
            st.markdown("#### Click the 'Record' button to start recording your voice.", unsafe_allow_html=True)

            col1, col2 = st.columns([2,3])

            with col1:
                st.write("")
            
            with col2:
            
                audio_bytes = audio_recorder(
                    text="",
                    recording_color="#ff0000",
                    neutral_color="#3a6883",
                    icon_name="microphone-lines",
                    icon_size="8x",
                )
            
            if audio_bytes:
                st.success("Audio recording successful!")

                # Save the recorded audio to a file
                with open("recorded_audio.wav", "wb") as f:
                    f.write(audio_bytes)

                # Transcribe the audio using OpenAI's Whisper ASR API
                with open("recorded_audio.wav", "rb") as f:
                    transcript = openai.Audio.translate(model="whisper-1", file=f, response_format="text")

                input_text = transcript
            else:
                input_text = ""  # Provide an empty input if audio recording is unsuccessful

        return input_text

    user_input = get_text()

    if user_input:
        user_keywords = extract_keywords(user_input)
        output = query(st.session_state.full_conversation.append({'role': 'user', 'content': user_input}))
        st.session_state.stored.append(user_input)
        st.session_state.prompted.append(output)
        unique_keywords.update(user_keywords)

        potential_friends = find_friend(unique_keywords)

        # Text-to-Speech
        st.audio(text_to_speech(output), format="audio/mp3")

    if st.session_state['prompted']:
        # Store user and bot messages separately
        all_user_messages = st.session_state['stored']
        all_bot_responses = st.session_state['prompted']

    # Filter out keywords
    conversation_keywords = [extract_keywords(message) for message in all_user_messages]
    for i in range(len(st.session_state['prompted']) - 1, -1, -1):
        message(st.session_state["prompted"][i], key=str(i))
        message(st.session_state['stored'][i], is_user=True, key=str(i) + '_user')

    # Display chat summary of all conversations
    display_chat_summary(all_user_messages)

    # Display keywords
    st.header(" ")
    st.markdown("#### Keywords", unsafe_allow_html=True)
    st.write("Keywords include:", ", ".join(unique_keywords))

    if potential_friends:
            st.header(" ")
            st.markdown("#### Potential Friends", unsafe_allow_html=True)
            for friend in potential_friends:
                st.write(f"**Name**: {friend['Name']}")
                st.write(f"**Matching Keywords**: {friend['Matching Keywords']}")
                st.subheader(" ")

# CONNECT
#if selected == "CONNECT":
#    st.header("CONNECT")
#    st.subheader("Connect with like-minded individuals.")

# FRIENDS
if selected == "FRIENDS":
    st.header("FRIENDS")
    st.subheader("Reach out to your newly made friends.")