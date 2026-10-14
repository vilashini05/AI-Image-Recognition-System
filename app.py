
import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Image Recognition",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ AI Image Recognition System")
st.write(
    "Upload an image and let AI predict what it contains."
)


@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):
        classifier = load_model()
        predictions = classifier(image, top_k=5)

    st.subheader("🔍 Prediction Results")

    best = predictions[0]

    st.success(f"Top prediction: {best['label']}")

    st.metric(
        "Model score",
        f"{best['score'] * 100:.2f}%"
    )

    df = pd.DataFrame(predictions)
    df["score_percent"] = df["score"] * 100

    st.subheader("Top 5 Predictions")

    st.dataframe(
        df[["label", "score_percent"]].rename(
            columns={
                "label": "Predicted label",
                "score_percent": "Score (%)"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    fig = px.bar(
        df,
        x="score_percent",
        y="label",
        orientation="h",
        title="Prediction Scores",
        labels={
            "score_percent": "Score (%)",
            "label": "Predicted label"
        }
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
