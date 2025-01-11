import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained vectorizer and model
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))


# Streamlit Interface Configuration
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"
if st.sidebar.button("ℹ️ About"):
    st.session_state.page = "About"
if st.sidebar.button("🌐 Social Media"):
    st.session_state.page = "Social Media"
    

# Display content based on the selected section
if st.session_state.page == "Home":
    st.title("📩 SMS Spam Detection System")
    st.markdown("#### A machine learning-powered tool to identify spam messages.")
    st.markdown("---")

    # Prediction Section
    st.markdown("### Enter the SMS below to check if it's Spam or Not Spam")
    input_sms = st.text_area("Type your SMS here:", placeholder="Enter your message...", height=150)

    if st.button("Predict"):
        if input_sms.strip() == "":
            st.warning("⚠️ Please enter a message to predict.")
        else:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tk.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Display Result
            st.markdown("---")
            if result == 1:
                st.error("🚨 **Spam Detected!**")
            else:
                st.success("✅ **This is Not Spam!**")

elif st.session_state.page == "About":
    st.title("About")
    st.markdown("""
        📜 **Project Description**  
        The SMS Spam Detection System is a machine learning-powered web application designed to classify SMS messages as either Spam or Not Spam. By leveraging Natural Language Processing (NLP) and supervised learning techniques, this system provides a quick and efficient way to filter unwanted messages.

        🚀 **Key Features**  
        - **User-Friendly Interface**: Built with Streamlit for an interactive and intuitive user experience.  
        - **Text Preprocessing**: Implements tokenization, stopword removal, and stemming for effective text analysis.  
        - **Accurate Predictions**: Trained using advanced machine learning algorithms to deliver precise classifications.  
        - **Real-Time Processing**: Instantly predicts the nature of an SMS after user input.  

        ⚙️ **Technologies Used**  
        - **Programming Language**: Python  
        - **Frameworks and Libraries**:  
            - **Streamlit**: For building the user interface.  
            - **NLTK**: For text preprocessing.  
            - **Scikit-learn**: For machine learning model development.  
        - **Tools**:  
            - **Pickle**: For saving and loading pre-trained models and vectorizers.  

        💡 **How It Works**  
        - **Input**: Users provide an SMS message through the web interface.  
        - **Text Transformation**: The input text undergoes preprocessing (lowercasing, tokenization, removal of stopwords, and stemming).  
        - **Vectorization**: The processed text is converted into numerical features using a pre-trained vectorizer.  
        - **Prediction**: A machine learning model classifies the message as Spam or Not Spam.  
        - **Output**: The result is displayed on the interface with a clear visual cue.  

        📈 **Applications**  
        - SMS filtering for personal or enterprise-level communication.  
        - Reducing unwanted promotional or phishing messages.  
        - Enhancing productivity by managing text-based communications effectively.  

        🛠 **Future Enhancements**  
        - Support for multiple languages in SMS detection.  
        - Mobile-friendly responsive design.  
        - Integration with email and messaging platforms.  

        ✨ **Contributors**  
        Mohammad Yusuf  
    """)

    
elif st.session_state.page == "Social Media":
    st.title("Follow Us")
    st.markdown("Connect with us on social media!")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://img.icons8.com/color/96/github.png", width=100)
        st.markdown("[GitHub](https://github.com/mohammadyusuf2)")
    with col2:
        st.image("https://img.icons8.com/color/96/linkedin-circled.png", width=100)
        st.markdown("[LinkedIn](https://www.linkedin.com/in/mohammad-yusuf-2b8b122a9/)")

st.markdown("---")
st.info("🔍 This tool uses natural language processing (NLP) techniques and machine learning to classify SMS messages.")
