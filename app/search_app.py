import os
import pandas as pd
import streamlit as st
from typing import List
from clip_model import CLIP, CLIPOpenAI
try:
    from glip import GLIP
    imported_glip = True
except Exception as e:
    print("Failed to import GLIP due to {}".format(e))
    imported_glip = False
from visualization import read_image, plot_bboxes
from datetime import datetime
from constants import DATA_DIR, INDEX_LOOKUP_FILE

DATA_SELECTION = {
    "0010_samples": "0010_samples.csv",
    "0100_samples": "0100_samples.csv",
    "0500_samples": "0500_samples.csv",
    "1000_samples": "1000_samples.csv",
}
MODEL_SELECTION = {
    "OpenAICLIP-FasterImage": CLIPOpenAI(INDEX_LOOKUP_FILE),
    "HuggingFaceCLIP": CLIP(),
    "OpenAICLIP": CLIPOpenAI(),
}

class Result:
    def __init__(self, image, score, glip_prediction=None) -> None:
        self.image = image
        self.score = score
        self.glip_prediction = glip_prediction


def read_csv(csv_name):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(DATA_DIR, x))
    return df


def get_results(df, clip_model, glip_model, query, score_thresh=20.0, top_k=3) -> List[Result]:
    results = []
    # use CLIP model to get similarity scores and pick the top_k
    df["score"] = clip_model.get_similarity_scores(df["image_path"].values, query)
    df = df.sort_values("score", ascending=False)
    df = df[df["score"] > score_thresh][:3]
    # use GLIP model to get bounding boxes
    for _, row in df.iterrows():
        image = read_image(row["image_path"])
        if glip_model is not None:
            glip_prediction = glip_model.predict(image, query)
        else:
            glip_prediction = None
        result = Result(image, row["score"], glip_prediction)
        results.append(result)
    return results


def show_results(results: List[Result], time_elapsed, top_k=3):
    top_k = min(top_k, len(results))
    st.write(
        f"Found {len(results)} results in {time_elapsed:.2f} seconds. Showing top {top_k} results below:")
    for result in results:
        image = result.image
        if result.glip_prediction is not None:
            image = plot_bboxes(
                image=image,
                bboxes=result.glip_prediction.bboxes,
                scores=result.glip_prediction.scores,
                labels=result.glip_prediction.labels,
            )
        caption = f"Score {result.score:.2f}"
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
    clip_model = MODEL_SELECTION[model_selection]
    df = read_csv(DATA_SELECTION[data_selection])
    if imported_glip:
        use_glip = st.checkbox("Use GLIP", value=True)
    glip_model = GLIP() if use_glip else None
    start_time = datetime.now()
    query = st.text_input(
        'Search Query', 'a roasted chicken')
    results = get_results(df, clip_model, glip_model, query)
    time_elapsed = datetime.now() - start_time

    show_results(results, time_elapsed.total_seconds())

if __name__ == "__main__":
    main()
