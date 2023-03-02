from datetime import datetime
import os
import pandas as pd

from clip_model import CLIP, CLIPOpenAI
# try:
#     from glip import GLIP
#     imported_glip = True
# except Exception as e:
#     print("Failed to import GLIP due to {}".format(e))
#     imported_glip = False
from constants import DATA_DIR, INDEX_LOOKUP_FILE



class BenchmarkLatency:
    def __init__(self, n_queries, n_runs) -> None:
        self.n_queries = n_queries
        self.n_runs = n_runs
        self.image_df = self._read_csv()
        self.queries = self._read_queries()
        self.models = {
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
            }
        }
        self.data = []
        self.output_file = "latency_benchmark.csv"

    def _read_csv(self, csv_name="0010_samples.csv"):
        df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
        df["image_path"] = df["image_path"].apply(lambda x: os.path.join(DATA_DIR, x))
        print(f"Read {len(df)} images from {csv_name}")
        return df

    def _read_queries(self, query_file="resources/queries.txt"):
        with open(query_file, "r") as f:
            queries = f.readlines()
        print(f"Read {len(queries)} queries from {query_file}")
        return queries

    def run(self):
        n_images = len(self.image_df)
        for model_name, model_config in self.models.items():
            model = model_config["model"](*model_config["args"])
            print(f"Running benchmark for {model_name}...")
            for query in self.queries[:3]:
                start = datetime.now()
                self._run_model(model, query)
                time = (datetime.now() - start).total_seconds()
                time_per_image = time / (n_images * self.n_runs)
                self._record(model_name, query, time_per_image)
            print(f"Finished benchmark for {model_name}...")

    def _run_model(self, model, query):
        for _ in range(self.n_runs):
            model.get_similarity_scores(self.image_df["image_path"].values, query)

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

def main(n_queries=10, n_runs=2):
    print("Benchmarking latency...")
    benchmark = BenchmarkLatency(n_queries=n_queries, n_runs=n_runs)
    benchmark.run()
    benchmark.save()


if __name__ == "__main__":
    main()
