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
    def __init__(self) -> None:
        self.image_df = self._read_csv()
        self.queries = self._read_queries()
        self.models = {
            "OpenAICLIP-FasterImage": CLIPOpenAI(INDEX_LOOKUP_FILE),
            "HuggingFaceCLIP": CLIP(),
            "OpenAICLIP": CLIPOpenAI(),
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

    def run(self, n=2):
        for model_name, model in self.models.items():
            print(f"Running benchmark for {model_name}...")
            for query in self.queries[:3]:
                for _ in range(n):
                    start = datetime.now()
                    model.get_similarity_scores(self.image_df["image_path"].values, query)
                    time = datetime.now() - start
                    self._record(model_name, query, time)
            print(f"Finished benchmark for {model_name}...")


    def _record(self, model_name, query, time):
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

def main():
    print("Benchmarking latency...")
    benchmark = BenchmarkLatency()
    benchmark.run()
    benchmark.save()


if __name__ == "__main__":
    main()
