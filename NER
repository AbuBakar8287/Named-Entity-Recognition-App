import streamlit as st
import spacy
from spacy import displacy
from streamlit_lottie import st_lottie
import requests

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation (you can replace with any other Lottie URL)
lottie_ner = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")

def main():
    # Page configuration
    st.set_page_config(page_title="NER App", page_icon="🧠", layout="centered")

    # Title and animation
    st.title("🧠 Named Entity Recognition (NER) App")
    st.markdown("Enhance your text analysis using spaCy and Streamlit!")

    if lottie_ner:
        st_lottie(lottie_ner, height=250)

    # Sidebar
    menu = ['🏠 Home', '📌 Named Entity Recognition']
    choice = st.sidebar.selectbox("🔧 Menu", menu)

    if choice == '🏠 Home':
        st.subheader("📝 Word Tokenization")
        raw_text = st.text_area("Enter text to tokenize", "Enter your text here...")
        if st.button("🔍 Tokenize"):
            doc = nlp(raw_text)
            tokens = [token.text for token in doc]
            st.success("✅ Tokenization Complete!")
            st.markdown("### Tokens:")
            st.write(tokens)

    elif choice == '📌 Named Entity Recognition':
        st.subheader("📊 Named Entity Recognition (NER)")
        raw_text = st.text_area("Enter text for NER", "Enter your text here...")
        if st.button("🚀 Analyze"):
            doc = nlp(raw_text)
            if doc.ents:
                st.success("✅ Named Entities Found!")
                st.markdown("### 📋 Entities and Labels")
                for ent in doc.ents:
                    st.markdown(
                        f"<div style='color:green; font-size:16px;'><strong>{ent.text}</strong> <span style='color:gray;'>({ent.label_})</span></div>",
                        unsafe_allow_html=True
                    )
                # Display using spaCy's Displacy visualizer
                html = displacy.render(doc, style="ent", jupyter=False)
                st.components.v1.html(html, scrolling=True, height=300)
            else:
                st.warning("⚠️ No named entities found.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;'>Created with ❤️ using <a href='https://spacy.io/' target='_blank'>spaCy</a> and <a href='https://streamlit.io/' target='_blank'>Streamlit</a></div>",
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
