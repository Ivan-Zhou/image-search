import os
import pandas as pd
import streamlit as st
from typing import List
from clip_model import CLIP, CLIPOpenAI, CLIPOpenAIFaiss

try:
    from glip import GLIP

    imported_glip = True
except Exception as e:
    print("Failed to import GLIP due to {}".format(e))
    imported_glip = False
from visualization import read_image, plot_bboxes
from datetime import datetime
from constants import DATA_DIR, INDEX_LOOKUP_FILE


if "DATA_SELECTION" not in st.session_state:
    st.session_state["DATA_SELECTION"] = {
        "0010_samples": "0010_samples.csv",
        "0100_samples": "0100_samples.csv",
        "0500_samples": "0500_samples.csv",
        "1000_samples": "1000_samples.csv",
        "5000_samples": "05000_samples.csv",
        "10000_samples": "10000_samples.csv",
    }

if "MODEL_SELECTION" not in st.session_state:
    st.session_state["MODEL_SELECTION"] = {
        "OpenAICLIP-FasterImage+FAISS": CLIPOpenAIFaiss(
            INDEX_LOOKUP_FILE, None, k_neighbors=5
        ),
        "OpenAICLIP-FasterImage": CLIPOpenAI(INDEX_LOOKUP_FILE),
        "HuggingFaceCLIP": CLIP(),
        "OpenAICLIP": CLIPOpenAI(),
    }

if "GLIP_MODEL" not in st.session_state:
    st.session_state["GLIP_MODEL"] = GLIP() if imported_glip else None


class Result:
    def __init__(self, image, score, glip_prediction=None) -> None:
        self.image = image
        self.score = score
        self.glip_prediction = glip_prediction


def read_csv(csv_name):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    df["image_path"] = df["image_path"].apply(lambda x: os.path.join(DATA_DIR, x))
    return df


def get_results(
    df, clip_model, glip_model, query, score_thresh=20.0, top_k=5
) -> List[Result]:
    results = []
    # use CLIP model to get similarity scores and pick the top_k
    df_output = clip_model.get_similarity_scores(df["image_path"].values, query)
    df_output = df_output.sort_values("score", ascending=False)
    df_output = df_output[df_output["score"] > score_thresh]
    # use GLIP model to get bounding boxes
    for _, row in df_output.iterrows():
        image = read_image(row["image_path"])
        if glip_model is not None:
            glip_prediction = glip_model.predict(image, query)
            if glip_prediction.n == 0:
                continue
        else:
            glip_prediction = None
        result = Result(image, row["score"], glip_prediction)
        results.append(result)
        if len(results) >= top_k:
            break
    return results


def show_results(results: List[Result], time_elapsed, top_k=3):
    top_k = min(top_k, len(results))
    st.write(
        f"Found {len(results)} results in {time_elapsed:.2f} seconds. Showing top {top_k} results below: "
    )
    count = 0
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
        count += 1
        if count >= top_k:
            break


def main():
    st.title("Image Search App")
    st.write("This app finds similar images to your query.")
    data_selection = st.selectbox(
        label="Dataset",
        options=st.session_state["DATA_SELECTION"].keys(),
    )
    model_selection = st.selectbox(
        label="Model",
        options=st.session_state["MODEL_SELECTION"].keys(),
    )
    clip_model = st.session_state["MODEL_SELECTION"][model_selection]
    df = read_csv(st.session_state["DATA_SELECTION"][data_selection])
    glip_model = st.session_state["GLIP_MODEL"]
    start_time = datetime.now()
    query = st.text_input("Search Query", "a roasted chicken")
    results = get_results(df, clip_model, glip_model, query)
    time_elapsed = datetime.now() - start_time

    show_results(results, time_elapsed.total_seconds())


if __name__ == "__main__":
    main()
