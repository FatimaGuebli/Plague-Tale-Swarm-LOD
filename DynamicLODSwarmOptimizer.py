from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DATA_FILE = "rat_swarm_telemetry.csv"
SHOW_VISUAL_VALIDATION = False
VISUAL_VALIDATION_PRESET = "Ultra"
KMEANS_MAX_ITER = 1

FEATURE_COLUMNS = [
    "distance_to_player",
    "in_view_frustum",
    "occlusion_factor",
    "light_intensity",
]

# Emphasize distance and visibility so the biased preset centroids
# produce a clearer hardware-quality separation.
FEATURE_WEIGHTS = np.array([3.0, 1.5, 1.0, 1.0])

LOD_NAMES = {
    0: "LOD_0_HERO",
    1: "LOD_1_SWARM",
    2: "LOD_2_CULL",
}

LOD_POLYGON_COSTS = {
    "LOD_0_HERO": 5000,
    "LOD_1_SWARM": 1000,
    "LOD_2_CULL": 50,
}

LOD_COLORS = {
    "LOD_0_HERO": "#d62828",
    "LOD_1_SWARM": "#f77f00",
    "LOD_2_CULL": "#577590",
}

QUALITY_PRESETS = {
    "Low": {
        "hardware_intent": "Focused on performance, with aggressive LOD transitions for weaker GPUs.",
        "initial_centroids": np.array(
            [
                [0.05, 1.00, 0.00, 1.00],  # Only extremely close rats remain heroes
                [0.15, 1.00, 0.50, 0.50],  # Mid-range starts very early
                [0.30, 0.00, 1.00, 0.00],  # Cull anything past 30% of max distance
            ]
        ),
    },
    "Medium": {
        "hardware_intent": "Balanced profile for mainstream hardware with moderate LOD transitions.",
        "initial_centroids": np.array(
            [
                [0.00, 1.00, 0.00, 1.00],
                [0.50, 1.00, 0.50, 0.50],
                [1.00, 0.00, 1.00, 0.00],
            ]
        ),
    },
    "High": {
        "hardware_intent": "Quality-oriented preset with later LOD reductions for stronger GPUs.",
        "initial_centroids": np.array(
            [
                [0.40, 1.00, 0.00, 1.00],  # Keep hero rats out to 40% distance
                [0.70, 1.00, 0.30, 0.70],  # Push the swarm tier further back
                [1.20, 0.00, 1.00, 0.00],  # Harder to reach the cull tier
            ]
        ),
    },
    "Ultra": {
        "hardware_intent": "Maximum fidelity for high-end GPUs, with conservative culling and wide spacing between LOD tiers.",
        "initial_centroids": np.array(
            [
                [0.80, 1.00, 0.00, 1.00],  # Almost everything stays hero quality
                [1.20, 1.00, 0.10, 0.90],  # Swarm tier moves to the extreme edge
                [2.00, 0.00, 1.00, 0.00],  # Distance-based culling is almost unreachable
            ]
        ),
    },
}


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Telemetry dataset not found: {data_path}")

    return pd.read_csv(data_path)


def scale_features(dataframe: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataframe[FEATURE_COLUMNS])


def apply_feature_weights(features: np.ndarray) -> np.ndarray:
    return features * FEATURE_WEIGHTS


def fit_lod_clusters(
    scaled_features: np.ndarray, initial_centroids: np.ndarray
) -> tuple[np.ndarray, KMeans]:
    weighted_features = apply_feature_weights(scaled_features)
    weighted_centroids = apply_feature_weights(initial_centroids)

    model = KMeans(
        n_clusters=3,
        init=weighted_centroids,
        n_init=1,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
    )
    cluster_ids = model.fit_predict(weighted_features)

    return cluster_ids, model


def build_lod_label_mapping(model: KMeans) -> dict[int, str]:
    centers = model.cluster_centers_ / FEATURE_WEIGHTS

    # Higher scores indicate a more visually important cluster:
    # closer, more visible, less occluded, and brighter.
    importance_scores = (
        (1.0 - centers[:, 0])
        + centers[:, 1]
        + (1.0 - centers[:, 2])
        + centers[:, 3]
    )
    ranked_cluster_ids = np.argsort(importance_scores)[::-1]

    return {
        int(ranked_cluster_ids[0]): "LOD_0_HERO",
        int(ranked_cluster_ids[1]): "LOD_1_SWARM",
        int(ranked_cluster_ids[2]): "LOD_2_CULL",
    }


def annotate_clusters(
    dataframe: pd.DataFrame,
    cluster_ids: np.ndarray,
    lod_label_mapping: dict[int, str],
) -> pd.DataFrame:
    annotated = dataframe.copy()
    cluster_labels = pd.Series(cluster_ids).map(lod_label_mapping)

    annotated["LOD_Cluster_ID"] = cluster_ids
    annotated["LOD_Cluster"] = pd.Categorical(
        cluster_labels,
        categories=list(LOD_NAMES.values()),
        ordered=True,
    )
    annotated["polygon_budget"] = cluster_labels.map(LOD_POLYGON_COSTS).astype(int)
    return annotated


def show_visual_validation(dataframe: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(11, 8))
    axis = cast(Axes3D, figure.add_subplot(111, projection="3d"))
    scatter_3d = cast(Any, axis.scatter)

    for lod_name in LOD_NAMES.values():
        subset = dataframe[dataframe["LOD_Cluster"] == lod_name]
        scatter_3d(
            xs=subset["distance_to_player"].to_numpy(),
            ys=subset["light_intensity"].to_numpy(),
            zs=subset["occlusion_factor"].to_numpy(),
            label=lod_name,
            color=LOD_COLORS[lod_name],
            alpha=0.65,
            s=20,
            edgecolors="none",
        )

    axis.set_title("Dynamic LOD Swarm Optimizer Validation")
    axis.set_xlabel("Distance To Player")
    axis.set_ylabel("Light Intensity")
    axis.set_zlabel("Occlusion Factor")
    axis.legend(loc="upper right")

    figure.tight_layout()
    if "agg" in plt.get_backend().lower():
        print()
        print("3D validation plot skipped because the current matplotlib backend is non-interactive.")
    else:
        plt.show()
    plt.close(figure)


def build_cluster_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = dataframe.groupby("LOD_Cluster", sort=False, observed=False).agg(
        rat_count=("LOD_Cluster", "size")
    )
    return summary.reindex(list(LOD_NAMES.values())).fillna(0).astype(int)


def build_polygon_report(dataframe: pd.DataFrame) -> dict[str, float]:
    baseline_polygons = len(dataframe) * LOD_POLYGON_COSTS["LOD_0_HERO"]
    optimized_polygons = int(dataframe["polygon_budget"].sum())
    polygons_saved = baseline_polygons - optimized_polygons
    reduction_percent = (polygons_saved / baseline_polygons) * 100

    return {
        "baseline_polygons": baseline_polygons,
        "optimized_polygons": optimized_polygons,
        "polygons_saved": polygons_saved,
        "reduction_percent": reduction_percent,
    }


def run_quality_preset(
    dataframe: pd.DataFrame,
    scaled_features: np.ndarray,
    preset_name: str,
    preset_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    cluster_ids, model = fit_lod_clusters(
        scaled_features, preset_config["initial_centroids"]
    )
    lod_label_mapping = build_lod_label_mapping(model)
    annotated = annotate_clusters(dataframe, cluster_ids, lod_label_mapping)
    cluster_summary = build_cluster_summary(annotated)
    polygon_report = build_polygon_report(annotated)

    print("=" * 72)
    print(f"Graphics Quality Preset: {preset_name}")
    print(f"Hardware Intent: {preset_config['hardware_intent']}")
    print()
    print("Cluster Summary (rat counts per LOD):")
    print(cluster_summary)
    print()
    print("Polygon Savings Report:")
    print(
        f"Baseline polygons (all rats at LOD_0_HERO): "
        f"{polygon_report['baseline_polygons']:,}"
    )
    print(f"Optimized polygons: {polygon_report['optimized_polygons']:,}")
    print(f"Polygon reduction: {polygon_report['polygons_saved']:,}")
    print(f"Reduction percent: {polygon_report['reduction_percent']:.2f}%")

    return annotated, polygon_report


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / DATA_FILE

    dataframe = load_dataset(data_path)
    scaled_features = scale_features(dataframe)
    comparison_rows: list[dict[str, str]] = []
    last_annotated: pd.DataFrame | None = None

    print("Dynamic LOD Swarm Optimizer")
    print(f"Source dataset: {data_path}")
    print("Results are shown during execution only and are not saved to disk.")
    print(f"Presets evaluated: {', '.join(QUALITY_PRESETS.keys())}")

    for preset_name, preset_config in QUALITY_PRESETS.items():
        annotated, polygon_report = run_quality_preset(
            dataframe, scaled_features, preset_name, preset_config
        )
        cluster_ids = annotated["LOD_Cluster_ID"].to_numpy()
        silhouette = silhouette_score(apply_feature_weights(scaled_features), cluster_ids)
        print(f"Silhouette Score: {silhouette:.4f}")

        comparison_rows.append(
            {
                "Preset": preset_name,
                "Reduction Percent": f"{polygon_report['reduction_percent']:.2f}%",
            }
        )
        if preset_name == VISUAL_VALIDATION_PRESET:
            last_annotated = annotated

    print()
    print("=" * 72)
    print("Final Comparison Table")
    comparison_table = pd.DataFrame(comparison_rows)
    print(comparison_table.to_string(index=False))

    if SHOW_VISUAL_VALIDATION and last_annotated is not None:
        print()
        print(f"Showing 3D validation plot for the {VISUAL_VALIDATION_PRESET} preset.")
        show_visual_validation(last_annotated)


if __name__ == "__main__":
    main()
