import joblib
import streamlit as st

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("spam_model.joblib")

model = load_model()

st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 2.5rem;
    }
    .stTextArea textarea {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📩 SMS Spam Detector")
st.caption("An NLP tool for classifying text messages")
st.write("Enter a message below to check whether it is spam or ham.")

message = st.text_area(
    "Message",
    placeholder="For example: Your mobile recharge was successful...",
    height=150
)

if st.button("Analyze message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        prediction = model.predict([message])[0]

        classifier = model.named_steps["classifier"]
        spam_index = list(classifier.classes_).index("spam")
        spam_score = model.predict_proba([message])[0][spam_index]

        if prediction == "spam":
            st.error("🚨 Result: Potential spam")
        else:
            st.success("✅ Result: Likely a normal message (ham)")

        st.progress(
            float(spam_score),
            text=f"Model spam score: {spam_score:.0%}"
        )
        st.caption("The score is the model’s estimate, not a guarantee.")

st.divider()
st.subheader("Model performance on test data")

accuracy_column, recall_column = st.columns(2)
accuracy_column.metric("Test accuracy", "96.1%")
recall_column.metric("Spam recall", "70.5%")

st.caption("These metrics are from the test run shown in your terminal.")

with st.expander("About this project"):
    st.write(
        "The model converts message text into TF-IDF features, "
        "then uses Multinomial Naive Bayes to classify it."
    )