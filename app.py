
   
   
import streamlit as st
from predict import predict_sentiment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Sentiment  Analyzer",
    page_icon=" 🧠",
    layout="centered"
)


st.markdown(
    """
    <div style="text-align:center;">
        <h1> Sentiment Analyzer</h1>
        <p style="color:gray;">Enter text and let AI detect the emotion behind it</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")


tweet = st.text_area(" Enter your text below", height=150)


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    analyze = st.button("  Analyze Sentiment")


if analyze:

    if not tweet.strip():
        st.warning("Please enter some text before analyzing.")
    else:
        with st.spinner("Analyzing sentiment... "):
            prediction, probability = predict_sentiment(tweet)

        # confidence extraction (safe for both formats)
        if isinstance(probability, (list, tuple)):
            confidence = max(probability)
        else:
            # confidence = probability if prediction == 1 else (1 - probability)
            confidence = float(probability[0][1])

        
        st.markdown("---")

        if prediction == 1:
            st.success("😊 Positive Sentiment Detected")
            st.balloons()
        else:
            st.error("😞 Negative Sentiment Detected")

        
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Prediction", "Positive 😊" if prediction == 1 else "Negative 😞")

        with col2:
            
            st.metric("Confidence", f"{float(confidence) * 100:.2f}%")

        
        st.progress(float(confidence))

        
        with st.expander("View Model Output"):
            st.write("Prediction:", prediction)
            st.write("Probability:", probability)


st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built with Streamlit • ML Sentiment Engine</p>",
    unsafe_allow_html=True
)

