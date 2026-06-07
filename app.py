import streamlit as st

from predict import predict_sentiment


st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="😊"
)

st.title("Twitter Sentiment Analysis")

tweet = st.text_area(
    "Enter Tweet"
)



if st.button("Analyze"):

    prediction, probability = predict_sentiment(tweet)

    if prediction == 0:

        st.success("Positive Sentiment 😊")

    else:

        st.error("Negative Sentiment 😞")

    confidence = probability if prediction == 1 else (1 - probability)

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )
