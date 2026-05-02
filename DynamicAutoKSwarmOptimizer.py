from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DATA_FILE = "rat_swarm_telemetry.csv"
SHOW_VISUAL_VALIDATION = False
KMEANS_MAX_ITER = 20
KMEANS_N_INIT = 5
MIN_K = 2
COARSE_SEARCH_POINTS = 12
REFINEMENT_RADIUS = 2
HERO_POLYGON_BUDGET = 5000
CULL_POLYGON_BUDGET = 50

FEATURE_COLUMNS = [
    "distance_to_player",
    "in_view_frustum",
    "occlusion_factor",
    "light_intensity",
]

# Distance and visibility matter most for LOD decisions, so they get
# slightly stronger influence during clustering.
FEATURE_WEIGHTS = np.array([3.0, 1.5, 1.0, 1.0])


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Telemetry dataset not found: {data_path}")

    return pd.read_csv(data_path)


def scale_features(dataframe: pd.DataFrame) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(dataframe[FEATURE_COLUMNS])


def apply_feature_weights(features: np.ndarray) -> np.ndarray:
    return features * FEATURE_WEIGHTS


def derive_search_ceiling(row_count: int) -> int:
    return min(row_count - 1, max(MIN_K + 4, int(np.sqrt(row_count))))


def derive_coarse_candidate_ks(row_count: int) -> list[int]:
    search_ceiling = derive_search_ceiling(row_count)
    fixed_ks = {2, 3, 4, 5, 6, search_ceiling}
    spaced_ks = {
        int(round(value))
        for value in np.geomspace(MIN_K, search_ceiling, num=COARSE_SEARCH_POINTS)
    }
    return sorted(
        candidate_k
        for candidate_k in fixed_ks | spaced_ks
        if MIN_K <= candidate_k <= search_ceiling
    )


def create_kmeans_model(cluster_count: int) -> KMeans:
    return KMeans(
        n_clusters=cluster_count,
        init="k-means++",
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
    )


def calculate_inertia_drop(previous_inertia: float | None, current_inertia: float) -> float:
    if previous_inertia is None or previous_inertia == 0.0:
        return 0.0

    return ((previous_inertia - current_inertia) / previous_inertia) * 100.0


def add_elbow_distances(profile_table: pd.DataFrame) -> pd.DataFrame:
    if len(profile_table) < 3:
        profile_table["Elbow Distance"] = 0.0
        return profile_table

    ks = profile_table["K"].to_numpy(dtype=float)
    inertias = profile_table["Inertia"].to_numpy(dtype=float)

    x_range = ks.max() - ks.min()
    y_range = inertias.max() - inertias.min()
    normalized_x = np.zeros_like(ks) if x_range == 0.0 else (ks - ks.min()) / x_range
    normalized_y = (
        np.zeros_like(inertias)
        if y_range == 0.0
        else (inertias - inertias.min()) / y_range
    )

    start_point = np.array([normalized_x[0], normalized_y[0]])
    end_point = np.array([normalized_x[-1], normalized_y[-1]])
    line_vector = end_point - start_point
    line_length = np.linalg.norm(line_vector)

    if line_length == 0.0:
        profile_table["Elbow Distance"] = 0.0
        return profile_table

    distances: list[float] = []
    for current_x, current_y in zip(normalized_x, normalized_y):
        point_vector = np.array([current_x, current_y]) - start_point
        cross_value = abs(
            (line_vector[0] * point_vector[1]) - (line_vector[1] * point_vector[0])
        )
        distances.append(float(cross_value / line_length))

    profile_table["Elbow Distance"] = distances
    return profile_table


def profile_coarse_k_range(weighted_features: np.ndarray) -> pd.DataFrame:
    candidate_ks = derive_coarse_candidate_ks(len(weighted_features))
    evaluation_rows: list[dict[str, int | float]] = []
    previous_inertia: float | None = None

    for candidate_k in candidate_ks:
        model = create_kmeans_model(candidate_k)
        model.fit(weighted_features)
        inertia_value = float(model.inertia_)
        evaluation_rows.append(
            {
                "K": int(candidate_k),
                "Inertia": inertia_value,
                "Inertia Drop %": calculate_inertia_drop(previous_inertia, inertia_value),
            }
        )
        previous_inertia = inertia_value

    profile_table = pd.DataFrame(evaluation_rows)
    profile_table = add_elbow_distances(profile_table)
    return profile_table


def detect_elbow_k(profile_table: pd.DataFrame) -> int:
    elbow_index = int(profile_table["Elbow Distance"].idxmax())
    return int(profile_table.loc[elbow_index, "K"])


def derive_refined_ks(elbow_k: int, row_count: int) -> list[int]:
    search_ceiling = derive_search_ceiling(row_count)
    minimum_k = max(MIN_K, elbow_k - REFINEMENT_RADIUS)
    maximum_k = min(search_ceiling, elbow_k + REFINEMENT_RADIUS)
    return list(range(minimum_k, maximum_k + 1))


def evaluate_refined_candidates(
    weighted_features: np.ndarray,
    candidate_ks: list[int],
    elbow_k: int,
) -> tuple[np.ndarray, KMeans, pd.DataFrame, int]:
    evaluation_rows: list[dict[str, int | float]] = []
    fitted_models: dict[int, tuple[np.ndarray, KMeans]] = {}
    previous_inertia: float | None = None

    for candidate_k in candidate_ks:
        model = create_kmeans_model(candidate_k)
        cluster_ids = model.fit_predict(weighted_features)
        inertia_value = float(model.inertia_)
        db_value = float(davies_bouldin_score(weighted_features, cluster_ids))
        ch_value = float(calinski_harabasz_score(weighted_features, cluster_ids))

        evaluation_rows.append(
            {
                "K": int(candidate_k),
                "Inertia": inertia_value,
                "Inertia Drop %": calculate_inertia_drop(previous_inertia, inertia_value),
                "Davies-Bouldin": db_value,
                "Calinski-Harabasz": ch_value,
                "Distance From Elbow": abs(candidate_k - elbow_k),
            }
        )
        fitted_models[int(candidate_k)] = (cluster_ids, model)
        previous_inertia = inertia_value

    refinement_table = pd.DataFrame(evaluation_rows)
    refinement_table["DB Rank"] = refinement_table["Davies-Bouldin"].rank(
        method="dense", ascending=True
    ).astype(int)
    refinement_table["CH Rank"] = refinement_table["Calinski-Harabasz"].rank(
        method="dense", ascending=False
    ).astype(int)
    refinement_table["Proximity Rank"] = refinement_table["Distance From Elbow"].rank(
        method="dense", ascending=True
    ).astype(int)
    refinement_table["Selection Score"] = (
        refinement_table["DB Rank"]
        + refinement_table["CH Rank"]
        + refinement_table["Proximity Rank"]
    )

    best_row = refinement_table.sort_values(
        by=[
            "Selection Score",
            "Distance From Elbow",
            "Davies-Bouldin",
            "Calinski-Harabasz",
            "K",
        ],
        ascending=[True, True, True, False, True],
    ).iloc[0]
    best_k = int(best_row["K"])
    refinement_table["Selected"] = np.where(refinement_table["K"] == best_k, "<--", "")

    best_cluster_ids, best_model = fitted_models[best_k]
    return best_cluster_ids, best_model, refinement_table, best_k


def build_cluster_metadata(
    model: KMeans,
) -> tuple[dict[int, str], dict[int, int], list[str], pd.DataFrame]:
    centers = model.cluster_centers_ / FEATURE_WEIGHTS
    importance_scores = (
        (1.0 - centers[:, 0])
        + centers[:, 1]
        + (1.0 - centers[:, 2])
        + centers[:, 3]
    )
    ranked_cluster_ids = np.argsort(importance_scores)[::-1]
    polygon_budgets = np.linspace(
        HERO_POLYGON_BUDGET,
        CULL_POLYGON_BUDGET,
        num=len(ranked_cluster_ids),
    ).round().astype(int)

    lod_label_mapping: dict[int, str] = {}
    polygon_budget_mapping: dict[int, int] = {}
    ordered_labels: list[str] = []

    for rank, cluster_id in enumerate(ranked_cluster_ids):
        if rank == 0:
            lod_label = f"LOD_{rank}_HERO"
        elif rank == len(ranked_cluster_ids) - 1:
            lod_label = f"LOD_{rank}_CULL"
        else:
            lod_label = f"LOD_{rank}_SWARM"

        lod_label_mapping[int(cluster_id)] = lod_label
        polygon_budget_mapping[int(cluster_id)] = int(polygon_budgets[rank])
        ordered_labels.append(lod_label)

    centroid_table = pd.DataFrame(
        {
            "LOD_Label": ordered_labels,
            "Importance Score": importance_scores[ranked_cluster_ids].round(4),
            "distance_to_player": centers[ranked_cluster_ids, 0].round(4),
            "in_view_frustum": centers[ranked_cluster_ids, 1].round(4),
            "occlusion_factor": centers[ranked_cluster_ids, 2].round(4),
            "light_intensity": centers[ranked_cluster_ids, 3].round(4),
            "polygon_budget": polygon_budgets,
        }
    )

    return lod_label_mapping, polygon_budget_mapping, ordered_labels, centroid_table


def annotate_clusters(
    dataframe: pd.DataFrame,
    cluster_ids: np.ndarray,
    lod_label_mapping: dict[int, str],
    polygon_budget_mapping: dict[int, int],
    ordered_labels: list[str],
) -> pd.DataFrame:
    annotated = dataframe.copy()
    cluster_id_series = pd.Series(cluster_ids)
    cluster_labels = cluster_id_series.map(lod_label_mapping)
    polygon_budgets = cluster_id_series.map(polygon_budget_mapping)

    annotated["LOD_Cluster_ID"] = cluster_ids
    annotated["LOD_Cluster"] = pd.Categorical(
        cluster_labels,
        categories=ordered_labels,
        ordered=True,
    )
    annotated["polygon_budget"] = polygon_budgets.astype(int)
    return annotated


def build_cluster_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    summary = (
        dataframe.groupby("LOD_Cluster", sort=False, observed=False)
        .agg(
            rat_count=("LOD_Cluster", "size"),
            avg_distance_to_player=("distance_to_player", "mean"),
            avg_occlusion_factor=("occlusion_factor", "mean"),
            avg_light_intensity=("light_intensity", "mean"),
            avg_polygon_budget=("polygon_budget", "mean"),
        )
        .round(4)
    )
    ordered_labels = list(dataframe["LOD_Cluster"].cat.categories)
    return summary.reindex(ordered_labels)


def build_polygon_report(dataframe: pd.DataFrame) -> dict[str, float]:
    baseline_polygons = len(dataframe) * HERO_POLYGON_BUDGET
    optimized_polygons = int(dataframe["polygon_budget"].sum())
    polygons_saved = baseline_polygons - optimized_polygons
    reduction_percent = (polygons_saved / baseline_polygons) * 100

    return {
        "baseline_polygons": baseline_polygons,
        "optimized_polygons": optimized_polygons,
        "polygons_saved": polygons_saved,
        "reduction_percent": reduction_percent,
    }


def show_visual_validation(dataframe: pd.DataFrame, ordered_labels: list[str]) -> None:
    figure = plt.figure(figsize=(11, 8))
    axis = cast(Axes3D, figure.add_subplot(111, projection="3d"))
    scatter_3d = cast(Any, axis.scatter)
    color_map = plt.get_cmap("tab20", len(ordered_labels))

    for index, lod_name in enumerate(ordered_labels):
        subset = dataframe[dataframe["LOD_Cluster"] == lod_name]
        scatter_3d(
            xs=subset["distance_to_player"].to_numpy(),
            ys=subset["light_intensity"].to_numpy(),
            zs=subset["occlusion_factor"].to_numpy(),
            label=lod_name,
            color=color_map(index),
            alpha=0.65,
            s=20,
            edgecolors="none",
        )

    axis.set_title("Dynamic Auto-K Swarm Optimizer Validation")
    axis.set_xlabel("Distance To Player")
    axis.set_ylabel("Light Intensity")
    axis.set_zlabel("Occlusion Factor")
    axis.legend(loc="upper right", fontsize=8)

    figure.tight_layout()
    if "agg" in plt.get_backend().lower():
        print()
        print("3D validation plot skipped because the current matplotlib backend is non-interactive.")
    else:
        plt.show()
    plt.close(figure)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / DATA_FILE

    dataframe = load_dataset(data_path)
    scaled_features = scale_features(dataframe)
    weighted_features = apply_feature_weights(scaled_features)

    coarse_profile = profile_coarse_k_range(weighted_features)
    elbow_k = detect_elbow_k(coarse_profile)
    refined_ks = derive_refined_ks(elbow_k, len(dataframe))
    best_cluster_ids, best_model, refinement_table, best_k = evaluate_refined_candidates(
        weighted_features,
        refined_ks,
        elbow_k,
    )
    lod_label_mapping, polygon_budget_mapping, ordered_labels, centroid_table = (
        build_cluster_metadata(best_model)
    )
    annotated = annotate_clusters(
        dataframe,
        best_cluster_ids,
        lod_label_mapping,
        polygon_budget_mapping,
        ordered_labels,
    )
    cluster_summary = build_cluster_summary(annotated)
    polygon_report = build_polygon_report(annotated)

    print("Dynamic Auto-K Swarm Optimizer")
    print(f"Source dataset: {data_path}")
    print("Results are shown during execution only and are not saved to disk.")
    print("K selection strategy: coarse elbow detection plus local multi-metric refinement.")
    print(f"Coarse candidate Ks: {derive_coarse_candidate_ks(len(dataframe))}")
    print()
    print("Coarse K Profiling (Inertia + Elbow Distance):")
    print(
        coarse_profile.round(
            {
                "Inertia": 2,
                "Inertia Drop %": 2,
                "Elbow Distance": 4,
            }
        ).to_string(index=False)
    )
    print()
    print(f"Elbow Candidate K: {elbow_k}")
    print(f"Refined K Window: {refined_ks}")
    print()
    print("Local K Validation (Davies-Bouldin + Calinski-Harabasz):")
    print(
        refinement_table.round(
            {
                "Inertia": 2,
                "Inertia Drop %": 2,
                "Davies-Bouldin": 4,
                "Calinski-Harabasz": 2,
            }
        ).to_string(index=False)
    )
    print()
    print(f"Selected K: {best_k}")
    print("Selection Basis: lowest combined rank from Davies-Bouldin, Calinski-Harabasz, and elbow proximity.")
    print()
    print("Dynamic Cluster Metadata:")
    print(centroid_table.to_string(index=False))
    print()
    print("Dynamic Cluster Summary:")
    print(cluster_summary)
    print()
    print("Polygon Savings Report:")
    print(f"Baseline polygons (all rats at hero quality): {polygon_report['baseline_polygons']:,}")
    print(f"Optimized polygons: {polygon_report['optimized_polygons']:,}")
    print(f"Polygon reduction: {polygon_report['polygons_saved']:,}")
    print(f"Reduction percent: {polygon_report['reduction_percent']:.2f}%")

    if SHOW_VISUAL_VALIDATION:
        show_visual_validation(annotated, ordered_labels)


if __name__ == "__main__":
    main()
