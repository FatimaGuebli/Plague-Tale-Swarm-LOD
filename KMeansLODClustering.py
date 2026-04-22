from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


DATA_FILE = "rat_swarm_telemetry.csv"
FEATURE_COLUMNS = [
    "distance_to_player",
    "in_view_frustum",
    "occlusion_factor",
    "light_intensity",
]

# Manual centroid seeds that represent the desired LOD tiers.
INITIAL_CENTROIDS = np.array(
    [
        [0.0, 1.0, 0.0, 1.0],  # LOD 0 (High): close, visible, clear, bright
        [0.5, 1.0, 0.5, 0.5],  # LOD 1 (Medium): mid-range, visible, partly hidden, dim
        [1.0, 0.0, 1.0, 0.0],  # LOD 2 (Low/Cull): far, off-screen, hidden, dark
    ]
)


def main() -> None:
    data_path = Path(__file__).resolve().parent / DATA_FILE
    dataframe = pd.read_csv(data_path)

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(dataframe[FEATURE_COLUMNS])

    kmeans = KMeans(
        n_clusters=3,
        init=INITIAL_CENTROIDS,
        n_init=1,
        max_iter=10,
        random_state=42,
    )

    dataframe["LOD_Cluster"] = kmeans.fit_predict(scaled_features)

    summary = (
        dataframe.groupby("LOD_Cluster", as_index=True)[
            ["distance_to_player", "light_intensity"]
        ]
        .mean()
        .sort_index()
    )

    print("LOD clustering summary:")
    print(summary)


if __name__ == "__main__":
    main()
