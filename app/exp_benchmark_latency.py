from datetime import datetime
import fire
import os
import pandas as pd

from clip_model import CLIP, CLIPOpenAI, CLIPOpenAIFaiss

try:
    from glip import GLIP

    imported_glip = True
except Exception as e:
    print("Failed to import GLIP due to {}".format(e))
    imported_glip = False
from constants import DATA_DIR, INDEX_LOOKUP_FILE
from visualization import read_image


DATA_SELECTION = {
    10: "0010_samples.csv",
    100: "0100_samples.csv",
    500: "0500_samples.csv",
    1000: "1000_samples.csv",
    5000: "05000_samples.csv",
    10000: "10000_samples.csv",
}


class BenchmarkLatency:
    def __init__(self, n_images, n_queries, n_runs) -> None:
        self.image_df = self._read_csv(n_images)
        self.n_queries = n_queries
        self.n_runs = n_runs
        self.queries = self._read_queries()
        self.models = {
            "OpenAICLIP-FasterImage+FAISS": {
                "model": CLIPOpenAIFaiss,
                "args": [INDEX_LOOKUP_FILE, None, 5],
            },
            "OpenAICLIP-FasterImage": {
                "model": CLIPOpenAI,
                "args": [INDEX_LOOKUP_FILE],
            },
            "OpenAICLIP": {
                "model": CLIPOpenAI,
                "args": [],
            },
            "HuggingFaceCLIP": {
                "model": CLIP,
                "args": [],
            },
        }
        if imported_glip:
            self.models["GLIP"] = {
                "model": GLIP,
                "args": [],
            }
        self.data = []
        self.output_file = "latency_benchmark.csv"

    def _read_csv(self, n_images):
        assert n_images in DATA_SELECTION, f"Invalid number of images {n_images}"
        csv_name = DATA_SELECTION[n_images]
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(DATA_DIR, x))
        print(f"Read {len(df)} images from {csv_name}")
        return df

    def _read_queries(self, query_file="resources/queries.txt"):
        with open(query_file, "r") as f:
            queries = f.readlines()
        queries = queries[: self.n_queries]
        print(f"Read {len(queries)} queries from {query_file}")
        return queries

    def run(self):
        n_images = len(self.image_df)
        for model_name, model_config in self.models.items():
            model = model_config["model"](*model_config["args"])
            print(f"Running benchmark for {model_name}...")
            for query in self.queries:
                start = datetime.now()
                for _ in range(self.n_runs):
                    try:
                        self._run_model(model_name, model, query)
                    except Exception as e:
                        print(f"Failed to run model {model_name} due to {e}")
                time = (datetime.now() - start).total_seconds()
                time_per_image = time / (n_images * self.n_runs)
                self._record(model_name, query, time_per_image)
            print(f"Finished benchmark for {model_name}...")

    def _run_model(self, model_name, model, query):
        if model_name == "GLIP":
            for _, row in self.image_df.iterrows():
                image = read_image(row["image_path"])
                model.predict(image, query)
        elif "CLIP" in model_name:
            model.get_similarity_scores(self.image_df["image_path"].values, query)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def _record(self, model_name, query, time):
        query = query.replace("\n", "")
        self.data.append(
            {
                "model": model_name,
                "query": query,
                "time": time,
            }
        )

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_file, index=False)
        print(f"Saved {len(df)} results to {self.output_file}")


def main(n_images=100, n_queries=20, n_runs=3):
    print("Benchmarking latency...")
    benchmark = BenchmarkLatency(n_images=n_images, n_queries=n_queries, n_runs=n_runs)
    benchmark.run()
    benchmark.save()


if __name__ == "__main__":
    fire.Fire(main)
