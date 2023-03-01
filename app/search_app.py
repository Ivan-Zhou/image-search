import os
import pandas as pd
import streamlit as st
from CLIP import CLIP, CLIPOpenAI
from visualization import read_image
from datetime import datetime
from constants import DATA_DIR, INDEX_LOOKUP_FILE

DATA_SELECTION = {
    "0010_samples": "0010_samples.csv",
    "0100_samples": "0100_samples.csv",
    "0500_samples": "0500_samples.csv",
    "1000_samples": "1000_samples.csv",
}
MODEL_SELECTION = {
    "HuggingFaceCLIP": CLIP(),
    "OpenAICLIP": CLIPOpenAI(),
    "OpenAICLIP-FasterImage": CLIPOpenAI(INDEX_LOOKUP_FILE),
}

def read_csv(csv_name):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(DATA_DIR, x))
    return df


def get_results(df, model, query, score_thresh=20.0) -> list:
    df["score"] = model.get_similarity_scores(df["image_path"].values, query)
    df = df.sort_values("score", ascending=False)
    df = df[df["score"] > score_thresh]
    return df[["image_path", "score"]].to_dict(orient="records")


def show_results(results: list, time_elapsed, top_k=3):
    top_k = min(top_k, len(results))
    st.write(
        f"Found {len(results)} results in {time_elapsed:.2f} seconds. Showing top {top_k} results below:")
    for result in results:
        image = read_image(result["image_path"])
        caption = f"Score {result['score']:.2f}"
        st.image(
            image,
            caption=caption,
            width=None,
            use_column_width=None,
            clamp=False,
            channels="RGB",
            output_format="auto",
        )


def main():
    st.title("Image Search App")
    st.write("This app finds similar images to your query.")
    data_selection = st.selectbox(
        label="Dataset",
        options=DATA_SELECTION.keys(),
    )
    model_selection = st.selectbox(
        label="Model",
        options=MODEL_SELECTION.keys(),
    )
    model = MODEL_SELECTION[model_selection]
    df = read_csv(DATA_SELECTION[data_selection])

    start_time = datetime.now()
    query = st.text_input(
        'Search Query', 'I was wearing a hat and eating food')
    results = get_results(df, model, query)
    time_elapsed = datetime.now() - start_time

    show_results(results, time_elapsed.total_seconds())

if __name__ == "__main__":
    main()
