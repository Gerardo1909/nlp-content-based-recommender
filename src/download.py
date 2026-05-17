if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("hf://datasets/mathigatti/spanish_imdb_synopsis/plots.csv")
    print(df.head())
    df.to_csv("data/plots.csv", index=False)
