"""Example runner script to test implementation visually."""

import random

import altair as alt
import pandas as pd

from kmeans import KMeans

if __name__ == "__main__":
    rng = random.Random(1)
    km = KMeans(rng, 5)
    points = [[rng.normalvariate() for _ in range(2)] for _ in range(100)]
    km.fit(points)

    centers_df = (
        pd.DataFrame.from_dict(km.cluster_centres, orient="index", columns=["x", "y"])
        .reset_index()
        .assign(name=lambda df_: "Cluster " + df_["index"].astype(str))
    )
    centers_chart = alt.Chart(centers_df).mark_point(
        color="black", filled=True, size=100
    ).encode(x="x", y="y") + alt.Chart(centers_df).mark_text(
        color="black", align="left"
    ).encode(x="x", y="y", text="name")
    pdf = pd.DataFrame(
        {
            "x": [i[0] for i in points],
            "y": [i[1] for i in points],
            "cluster": [str(i[1]) for i in km._datapoints],
        }
    )
    c = alt.Chart(pdf).mark_point().encode(x="x", y="y", color="cluster")
    (c + centers_chart).save("test_chart.png")
