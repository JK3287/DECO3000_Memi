# streamlit run '/Users/yungvenuz/Documents/Uni/Year 3 DC/DECO3000/DECO3000_Memi/a.py'

# I like pancakes fishing and window shopping
# I also fear graduation

# IMPORTS

# Visual Studio - Version 1.76.1
# Streamlit and relevant plug-ins for website design
import streamlit as st # Version 1.27.2
from streamlit_option_menu import option_menu # Version 0.3.6
from streamlit_chat import message # Version 0.1.1
from audio_recorder_streamlit import audio_recorder # Version 0.0.8

# .env and relevant plug-ins
from dotenv import load_dotenv # Version 1.0.0
import os # Python Version 2.7.18

# OpenAI and relevant plug-ins for Large Language Model
import openai # Version 0.28.1
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from gtts import gTTS # Version 2.4.0
from io import BytesIO # Python Version 2.7.18

# Spacy and relevant plug-ins
# type in 'spacy download en_core_web_sm'
import spacy # Version 3.7.2
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
import pandas as pd # Version 1.5.3

# PROGRAM STARTS HERE!

load_dotenv() # Load OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config( # Load Streamlit Website
    page_title="Chatbot",
    page_icon=":robot:"
) 


st.markdown( # CSS Markdown Rules
    """
<style>
* {
   font-family: Montserrat;
   text-align: center;
} 

.stRadio [role=radiogroup]{
        align-items: center;
        justify-content: center;
        transform: scale(1.625);
    }

</style>
""",
    unsafe_allow_html=True,
) 

potential_friends = [] # Friends made from an exchange of dialogue
potential_friends_over_time = [] # Storing friends made along the conversation
user_keywords = set() # User keywords

if 'full_conversation' not in st.session_state: # Prompt template
        st.session_state['full_conversation'] = [{'role': 'system',
                                                  'content': 'You are a chatbot named Memi. I want you to remind me, an elderly human, to look at the bottom of the page to find a list of potential friends once I talk enough about myself when I want to make a friend, otherwise I want you to assist me with creating a conversation that looks into past positive memories to put me in a better mood.  I want you to inquire about details of the memories I am telling you about, so we can get a better understanding of what happened.  Start with asking me about what I want to talk about, but if I am not sure, please prompt me with a conversation starter to help.  Please keep the conversation in a positive light, and help me to appreciate my past experiences. Do not make answers longer than 50 words. Your tone is friendly and casual, yet caring and empathetic.'}]
        
if 'selected_friends' not in st.session_state: # Ensure friends are selected
    st.session_state.selected_friends = [] 

with st.sidebar: # Menu
    selected = option_menu(
        menu_title="Memi",  
        options=["TALK", "FRIENDS", "CHATLOG"],  
    ) 

def summarize_text(chat_log): # Summarize
        llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
        docs = [Document(page_content=t) for t in chat_log]
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        summary = chain.run(docs)
        return summary

# TALK Page - Communicate with chatbot
if selected == "TALK":

    st.markdown("<h1 style='font-size: 84px;'>Memi</h1>", unsafe_allow_html=True)  # Headers and picture
    st.markdown("<h2 style='font-size: 48px;'>- TALK -</h2>", unsafe_allow_html=True)
    st.markdown("<img src='https://i.ibb.co/3Sdk7zr/memi-talk.png' alt='Memi Talk' style='width:50%;'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='font-weight: normal;'>Feel free to talk with <b>Memi</b> as an <b>AI chatbot</b>, providing <b>reminiscence</b> therapy.</h3>",
        unsafe_allow_html=True
    )
    st.header(" ")
    st.header(" ")
    unique_keywords = set()

    if 'prompted' not in st.session_state:  # Storing GPT-3.5 responses
        st.session_state['prompted'] = []

    if 'stored' not in st.session_state:  # Storing user responses
        st.session_state['stored'] = []

    if 'conversation_keywords' not in st.session_state:  # Find keywords
        st.session_state['conversation_keywords'] = []

    def text_to_speech(text):  # Text to speech function
        audio_bytes = BytesIO()
        tts = gTTS(text=text, lang="en", slow=False)
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()

    def query(user_text):  # Response function
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state.full_conversation
        )
        message = response.choices[0].message.content
        st.session_state.full_conversation.append({'role': 'system', 'content': message})
        bot_keywords = extract_keywords(message, 'bot')
        st.session_state.conversation_keywords.extend(bot_keywords)
        return message

    def extract_keywords(text, role):
        if role == 'user':
            doc = nlp(text)
            keywords = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
            unique_keywords.update(keywords)
            return keywords
        else:
            return []

    def display_chat_summary(chat_log):  # Display
        st.header(" ")
        st.header(" ")
        st.markdown("## Chat Summary", unsafe_allow_html=True)
        summary = summarize_text(chat_log)
        st.markdown(f"<p style='font-size: 24px; font-family: Montserrat;'>{summary}</p>", unsafe_allow_html=True)

    def find_friend(unique_keywords):  # Find friends
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

    if 'last_saved_index' not in st.session_state:
        st.session_state.last_saved_index = 0

    def save_chat():
        last_saved_index = st.session_state.last_saved_index
        saved_chat = {
            'user_messages': st.session_state.stored[last_saved_index:],
            'bot_responses': st.session_state.prompted[last_saved_index:],
        }

        if 'saved_chats' not in st.session_state:
            st.session_state.saved_chats = []

        st.session_state.saved_chats.append(saved_chat)
        st.session_state.last_saved_index = len(st.session_state.stored)
        st.success("Chat saved successfully!")

    all_user_messages = []  # Initialize the list

    def get_text():  # Input function
        st.markdown("<h2 style='font-size: 36px;'>Choose Input</h2>", unsafe_allow_html=True)
    
        # Add "None" to the options
        input_option = st.radio("", ("Text", "Record Voice", "Stop"))

        if input_option == "Text":
            st.header(" ")
            st.header(" ")
            st.markdown(
                "<h3 style='font-weight: normal;'>Please type in the <b>box</b> below and press <b>Enter</b> to chat with <b>Memi</b>. </h3>",
                unsafe_allow_html=True
            )

            st.markdown("<style>input[type='text'] { font-size: 24px; }</style>", unsafe_allow_html=True)

            # Check for the presence of 'mark_for_submit' in the session state
            if 'mark_for_submit' not in st.session_state:
                st.session_state.mark_for_submit = False

            with st.form("my_form"):
                user_input = st.text_input(" ", "", key="user_input")

                # Use form_submit_button directly without the on_click parameter
                submitted = st.form_submit_button(label="Submit")

            # Handle the form submission
            if submitted:
                st.session_state.mark_for_submit = True

            return user_input  # Return user_input directly when "Text" is selected

        elif input_option == "Record Voice":
            st.header(" ")
            st.header(" ")
            st.markdown(
                "<h3 style='font-weight: normal;'>Please click on the <b>microphone</b> below to record your voice so <b>Memi</b> can listen. <b>Stop</b> speaking when you are done. </h3>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns([2, 3])

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

                with open("recorded_audio.wav", "wb") as f:
                    f.write(audio_bytes)  # Save the recorded audio to a file

                with open("recorded_audio.wav", "rb") as f:
                    transcript = openai.Audio.translate(model="whisper-1", file=f, response_format="text")  # Transcribe the audio using OpenAI's Whisper ASR API
                return transcript  # Return transcript when "Record Voice" is selected
        else: 
            st.header(" ")
            st.header(" ")
        return ""  # Return an empty string if "None" is selected

    user_input = get_text()  # Obtain text input from user

    if user_input:
        user_keywords = extract_keywords(user_input, 'user')
        output = query(st.session_state.full_conversation.append({'role': 'user', 'content': user_input}))
        st.session_state.stored.append(user_input)
        st.session_state.prompted.append(output)

        st.header(" ")
        st.header(" ")
        st.markdown("<h2 style='font-size: 36px;'> Your Conversation </h2>", unsafe_allow_html=True)
        st.markdown(
        "<h3 style='font-weight: normal;'>Click on the <b>triangle</b> button of the <b>white</b> audio player to hear <b>Memi</b>.</h3>",
        unsafe_allow_html=True
    )
        st.audio(text_to_speech(output), format="audio/mp3")  # Text-to-Speech

        st.markdown(
        "<h3 style='font-weight: normal;'>Click the <b>top</b> of the <b>chat</b> to toggle visibility.</h3>",
        unsafe_allow_html=True
    )
    if st.session_state['prompted']:  # Store user and bot messages separately

        all_user_messages = st.session_state['stored']
        all_bot_responses = st.session_state['prompted']

    conversation_keywords = [extract_keywords(message, 'user') for message in all_user_messages]  # Filter out keywords

    expander_expanded = st.expander("SHOW/HIDE", expanded=True)
    with expander_expanded:
        for i in range(len(st.session_state['prompted']) - 1, -1, -1):
            user_message = st.session_state["stored"][i]
            bot_response = st.session_state["prompted"][i]

            user_keywords = extract_keywords(user_message, 'user')
            bot_keywords = extract_keywords(bot_response, 'bot')

            message(f'<p style="font-size:24px; font-family: Montserrat;">{st.session_state["prompted"][i]}</p>', allow_html=True, logo="https://i.ibb.co/ykTNV5z/icon-bot.png", key=str(i))
            message(f'<p style="font-size:24px; font-family: Montserrat;">{st.session_state["stored"][i]}</p>', allow_html=True, is_user=True, logo="https://i.ibb.co/6nCCgkt/icon-user.png", key=str(i) + '_user')

    unique_keywords.update(user_keywords)
    potential_friends = find_friend(unique_keywords)
    potential_friends_over_time.append(potential_friends)

    st.header(" ")
    st.header(" ")

    st.markdown(  # Add a button to save the chat with custom styling
        """
        <style>
            .large-button {
                width: 200%;
                padding: 10px; /* You can adjust the padding as needed */
                font-size: 18px; /* You can adjust the font size as needed */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 style='font-size: 36px;'>Save Chat</h2>", unsafe_allow_html=True)
    if st.button("Click to **save** your **conversation** on **CHATLOG**", on_click=save_chat):
        pass  # Add any additional functionality here if needed

    chat_summary = summarize_text(all_user_messages)  # Display chat summary of all conversations
    st.session_state['chat_summary'] = chat_summary
    st.header(" ")
    st.header(" ")

    st.markdown("<h2 style='font-size: 36px;'>Keywords Recorded</h2>", unsafe_allow_html=True) # Display keywords
    st.markdown(
        f"<p style='font-size: 24px; font-family: Montserrat;'>Keywords include: {', '.join(unique_keywords)}</p>",
        unsafe_allow_html=True
    )

    if potential_friends_over_time:  # Show potential friends made through the full conversation

        st.header(" ")
        st.header(" ")
        st.markdown("<h2 style='font-size: 36px;'>Find Friends</h2>", unsafe_allow_html=True) # Display keywords

        for i, friends_list in enumerate(potential_friends_over_time):

            for friend in friends_list:

                st.markdown(  # Display name and matching keywords
                    f"<p style='font-size: 24px; font-family: Montserrat;'><strong>Name:</strong> {friend['Name']}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat;'><strong>Matching Keywords:</strong> {friend['Matching Keywords']}</p>",
                    unsafe_allow_html=True
                )

                if st.button(f"Click to add **{friend['Name']}** as a friend"):  # Button to add the friend to the "FRIENDS" page
                    st.session_state.selected_friends.append(friend)
                st.subheader(" ")

    st.subheader(" ")

# FRIENDS Page - Find friends that were added from a conversation
if selected == "FRIENDS":

    st.markdown("<h1 style='font-size: 84px;'>Memi</h1>", unsafe_allow_html=True)  # Headers and picture
    st.markdown("<h2 style='font-size: 48px;'>- FRIENDS -</h2>", unsafe_allow_html=True)
    st.markdown("<img src='https://i.ibb.co/7jMJPLZ/memi-friends.png' alt='Memi Friends' style='width:50%;'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='font-weight: normal;'>With the assistance of <b>Memi</b>, reach out to your newly made <b>friends</b> and never feel alone ever <b>again</b>.</h3>",
        unsafe_allow_html=True
    )
    st.header(" ")
    st.header(" ")
    st.header(" ")

    friends = st.session_state.selected_friends  # Display all friends added in a grid

    i = 0  # Reset i before starting the loop

    for i in range(0, len(friends), 2):  # Create a grid layout for the friends
        columns = st.columns(2)

        for col in columns:
            if i < len(friends):
                friend = friends[i]

                col.markdown(  # You can use a custom profile picture here
                    f'<p style="text-align:center;"><img src="https://i.ibb.co/wc8cbvv/memi-face.png" alt="User Icon" width="100"></p>',
                    unsafe_allow_html=True
                )
                col.markdown(
                    f"<p style='font-size: 24px; font-family: Montserrat; text-align: center;'><strong>{friend['Name']}</strong></p>",
                    unsafe_allow_html=True
                )
                col.markdown(
                    f"<p style='font-size: 16px; font-family: Montserrat; text-align: center;'><strong>Matching Keywords:</strong> {friend['Matching Keywords']}</p>",
                    unsafe_allow_html=True
                )

            i += 1  # Move the incrementation inside the if block to avoid unnecessary increments

    if i % 2 != 0:  # Check if there's an odd number of friends to display
        st.columns([st.empty(), st.empty()])  # Add an empty column to maintain the layout

        
# CHATLOG Page - Keep a record of your conversation with the chatbot
if selected == "CHATLOG":

    st.header(" ")
    st.header(" ")

    st.markdown("<h1 style='font-size: 84px;'>Memi</h1>", unsafe_allow_html=True) # Headers and picture 
    st.markdown("<h2 style='font-size: 48px;'>- CHATLOG -</h2>", unsafe_allow_html=True)
    st.markdown("<img src='https://i.ibb.co/RSThqGL/memi-chatlog.png' alt='Memi Friends' style='width:50%;'>", unsafe_allow_html=True)
    st.markdown(
        "<h3 style='font-weight: normal;'><b>Read</b> through your <b>saved</b> conversations. Click on the <b>top</b> of each <b>chat</b> to toggle visibility.</h3>",
        unsafe_allow_html=True
    )

    st.header(" ")
    st.header(" ")

    if 'saved_chats' in st.session_state and st.session_state.saved_chats:
        for idx, saved_chat in reversed(list(enumerate(st.session_state.saved_chats))):
            st.markdown(f"<h2 style='font-size: 36px;'>Chat {idx + 1}</h2>", unsafe_allow_html=True)
            with st.expander(f"SHOW/HIDE", expanded=True):
                st.markdown("### Chat Summary", unsafe_allow_html=True)  # Show chat summary

                # Extract chat summary based on the current saved chat
                chat_summary = summarize_text(saved_chat['user_messages'])
                st.markdown(f"<p style='font-size: 24px; font-family: Montserrat;'>{chat_summary}</p>", unsafe_allow_html=True)

                for i in range(len(saved_chat['user_messages'])):
                    # Display user_messages and bot_responses directly
                    st.markdown("<h3 style='font-size: 24px;'>USER:</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='text-align:center;'><img src='https://i.ibb.co/vzKbVCv/user-circle.png' alt='User Icon' width='100'></p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='font-size: 24px; font-family: Montserrat; text-align: center;'>{saved_chat['user_messages'][i]}</p>",
                        unsafe_allow_html=True
                    )
                    st.header(" ")

                    st.markdown("<h3 style='font-size: 24px;'>MEMI:</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f"<p style='text-align:center;'><img src='https://i.ibb.co/TwRV1hf/bot-circle.png' alt='Bot Icon' width='100'></p>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<p style='font-size: 24px; font-family: Montserrat; text-align: center;'>{saved_chat['bot_responses'][i]}</p>",
                        unsafe_allow_html=True
                    )
            st.header(" ")
            