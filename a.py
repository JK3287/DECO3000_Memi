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
        transform: scale(2);
    }

</style>
""",
    unsafe_allow_html=True,
)

# MENU

potential_friends = []
potential_friends_over_time = []
user_keywords = set()

if 'selected_friends' not in st.session_state:
    st.session_state.selected_friends = []

with st.sidebar:
    selected = option_menu(
        menu_title="Memi",  # required
        options=["TALK", "FRIENDS", "CHATLOG"],  # required
    )

# TALK CHATBOT
if selected == "TALK":
    
    # Header
    st.markdown("<h1 style='font-size: 120px;'>TALK</h1>", unsafe_allow_html=True)
    st.header(" ")
    st.header(" ")
    st.markdown(
    "<h2 style='font-weight: normal;'>Feel free to talk to an <b>AI chatbot</b>, who will provide you with <b>reminiscence</b> therapy.</h2>",
    unsafe_allow_html=True
)
    st.header(" ")
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
                                                  'content': 'You are a chatbot named Memi. I want you to remind me, a human, to look at the bottom of the page to find a list of potential friends once I talk enough about myself when I want to make a friend, otherwise I want you to assist me with creating a conversation that looks into past positive memories to put me in a better mood.  I want you to inquire about details of the memories I am telling you about, so we can get a better understanding of what happened.  Start with asking me about what I want to talk about, but if I am not sure, please prompt me with a conversation starter to help.  Please keep the conversation in a positive light, and help me to appreciate my past experiences. Do not make answers longer than 50 words. Your tone is friendly and casual, yet caring and empathetic.'}]

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
        keywords = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
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
        st.header(" ")
        st.markdown("## Chat Summary", unsafe_allow_html=True)
        summary = summarize_text(chat_log)
        st.markdown(f"<p style='font-size: 24px; font-family: Montserrat;'>{summary}</p>", unsafe_allow_html=True)

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

     # Save Chat function
    def save_chat():
        saved_chat = {
            'user_messages': st.session_state.stored.copy(),
            'bot_responses': st.session_state.prompted.copy(),
        }

        if 'saved_chats' not in st.session_state:
            st.session_state.saved_chats = []
        st.session_state.saved_chats.append(saved_chat)
        st.success("Chat saved successfully!")

    # Initialize the list
    all_user_messages = []

    # Input function
    def get_text():
        st.markdown("## Choose input method:", unsafe_allow_html=True)
        input_option = st.radio("", ("Text", "Record Voice"))

        if input_option == "Text":
            st.header(" ")
            st.header(" ")
            st.markdown("## Please type in the box below.", unsafe_allow_html=True)
            st.markdown("<style>input[type='text'] { font-size: 36px; }</style>", unsafe_allow_html=True)
            user_input = st.text_input("", " ")
            return user_input  # Return user_input directly when "Text" is selected
        else:
            # Record audio using audio_recorder ONLY WORKS IF TEXT IS FIRST
            st.header(" ")
            st.header(" ")
            st.markdown("## Click the 'Record' button to start recording your voice.", unsafe_allow_html=True)

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
                return transcript  # Return transcript when "Record Voice" is selected

        return ""  # Return an empty string if neither option is selected

    user_input = get_text()

    if user_input:
        user_keywords = extract_keywords(user_input)
        output = query(st.session_state.full_conversation.append({'role': 'user', 'content': user_input}))
        st.session_state.stored.append(user_input)
        st.session_state.prompted.append(output)

        

        # Text-to-Speech
        st.audio(text_to_speech(output), format="audio/mp3")

    if st.session_state['prompted']:
        # Store user and bot messages separately
        all_user_messages = st.session_state['stored']
        all_bot_responses = st.session_state['prompted']

    # Filter out keywords
    conversation_keywords = [extract_keywords(message) for message in all_user_messages]
    
    for i in range(len(st.session_state['prompted']) - 1, -1, -1):
        message(f'<p style="font-size:24px; font-family: Montserrat;">{st.session_state["prompted"][i]}</p>', allow_html=True, key=str(i))
        message(f'<p style="font-size:24px; font-family: Montserrat;">{st.session_state["stored"][i]}</p>', allow_html=True, is_user=True, key=str(i) + '_user')

    unique_keywords.update(user_keywords)
    potential_friends = find_friend(unique_keywords)
    potential_friends_over_time.append(potential_friends)

    # Display chat summary of all conversations
    chat_summary = summarize_text(all_user_messages)
    st.session_state['chat_summary'] = chat_summary

    # Display keywords
    st.header(" ")
    st.header(" ")
    st.markdown("## Keywords", unsafe_allow_html=True)
    st.markdown(
    f"<p style='font-size: 24px; font-family: Montserrat;'>Keywords include: {', '.join(unique_keywords)}</p>",
    unsafe_allow_html=True
)

    if potential_friends_over_time:
        st.header(" ")
        st.header(" ")
        st.markdown("## Potential Friends", unsafe_allow_html=True)

        for i, friends_list in enumerate(potential_friends_over_time):
            st.markdown(f"### Conversation {i + 1}")
            for friend in friends_list:
                # Button to add the friend to the "FRIENDS" page
                if st.button(f"ADD {friend['Name']}"):
                    # Pass the friend's information to the "FRIENDS" page
                    st.session_state.selected_friends.append(friend)

                st.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><strong>Name:</strong> {friend['Name']}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><strong>Matching Keywords:</strong> {friend['Matching Keywords']}</p>",
                    unsafe_allow_html=True
                )

    st.subheader(" ")

    # Add a button to save the chat
    st.button("Save Chat", on_click=save_chat)


# FRIENDS
if selected == "FRIENDS":
    st.markdown("<h1 style='font-size: 120px;'>FRIENDS</h1>", unsafe_allow_html=True)
    st.header(" ")
    st.header(" ")
    st.markdown(
    "<h2 style='font-weight: normal;'><b>Reach</b> out to your newly made <b>friends</b>.</h2>",
    unsafe_allow_html=True
)

    st.header(" ")
    st.header(" ")
    # Display all friends added in a grid
    friends = st.session_state.selected_friends

    # Create a grid layout for the friends
    for i in range(0, len(friends), 2):
        columns = st.columns(2)

        for col in columns:
            if i < len(friends):
                friend = friends[i]

                # You can use a custom profile picture here
                emoji_html = "<span style='font-size: 4em;'>😄</span>"
                col.markdown(emoji_html, unsafe_allow_html=True)
                col.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><strong>{friend['Name']}</strong></p>",
                    unsafe_allow_html=True
                )
                col.markdown(
                    f"<p style='font-size: 16px; font-family: Montserrat;'><strong>Matching Keywords:</strong> {friend['Matching Keywords']}</p>",
                    unsafe_allow_html=True
                )

            i += 1

        # Apply CSS for 50% width
        col1, col2 = columns
        col1.markdown("<style>div.css-1l02z8t { width: 50% !important; }</style>", unsafe_allow_html=True)
        col2.markdown("<style>div.css-1l02z8t { width: 50% !important; }</style>", unsafe_allow_html=True)
        
# CHATLOG
if selected == "CHATLOG":
    st.markdown("<h1 style='font-size: 120px;'>CHATLOG</h1>", unsafe_allow_html=True)
    st.header(" ")
    st.header(" ")
    st.markdown(
        "<h2 style='font-weight: normal;'>Here is the chatlog.</h2>",
        unsafe_allow_html=True
    )

    # Display saved chats
    if 'saved_chats' in st.session_state and st.session_state.saved_chats:
        for idx, saved_chat in enumerate(st.session_state.saved_chats):
            st.markdown(f"### Chat {idx + 1}")
            
            st.header(" ")
            st.header(" ")

            st.markdown("## Chat Summary", unsafe_allow_html=True)

            st.header(" ")
            st.header(" ")

            # Display stored chat summary
            chat_summary = st.session_state.get('chat_summary', '')
            st.markdown(f"<p style='font-size: 24px; font-family: Montserrat;'>{chat_summary}</p>", unsafe_allow_html=True)

            # Display user_messages and bot_responses directly
            for i in range(len(saved_chat['user_messages'])):
                st.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><b>User:</b> {saved_chat['user_messages'][i]}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><b>Memi:</b> {saved_chat['bot_responses'][i]}</p>",
                    unsafe_allow_html=True
                )

            st.subheader(" ")