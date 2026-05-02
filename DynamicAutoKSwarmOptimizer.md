# DynamicAutoKSwarmOptimizer.py

## Line-by-Line Explanation

This document explains [DynamicAutoKSwarmOptimizer.py](/C:/Projects/Plague_Tale_LOD/Plague-Tale-Swarm-LOD/DynamicAutoKSwarmOptimizer.py) in code order.

The main improvement in this version is that it no longer selects `K` with a brute-force silhouette sweep. Instead, it uses a more detailed two-stage strategy:

- a **coarse elbow search** on inertia
- a **local refinement step** using Davies-Bouldin and Calinski-Harabasz scores

## Lines 1-11: Imports

- **Line 1** imports `Path` for safe filesystem paths.
- **Line 2** imports `Any` and `cast` for typing support in the 3D plotting section.
- **Lines 4-5** import `numpy` and `pandas` for numeric work and dataframe handling.
- **Line 6** imports `KMeans`, the clustering model used by the script.
- **Line 7** imports `calinski_harabasz_score` and `davies_bouldin_score`, which are now the main validation metrics for choosing `K`.
- **Line 8** imports `MinMaxScaler` to normalize the dataset.
- **Line 10** imports `matplotlib.pyplot` for plotting.
- **Line 11** imports `Axes3D` for 3D scatter visualization.

## Lines 14-22: Global Constants

- **Line 14** defines the CSV dataset file name.
- **Line 15** controls whether the validation plot should be shown.
- **Line 16** sets the maximum number of iterations for K-Means.
- **Line 17** sets `n_init=5`, which makes K-Means more stable than a single random start.
- **Line 18** sets the smallest valid cluster count to `2`.
- **Line 19** sets how many coarse `K` points are sampled during the first-stage search.
- **Line 20** sets how many neighboring `K` values are checked around the detected elbow.
- **Lines 21-22** define the hero and cull polygon budgets used in the optimization report.

## Lines 24-33: Features and Feature Weighting

- **Lines 24-29** define the four telemetry columns used for clustering.
- **Lines 31-32** explain that distance and visibility are more important than the other features for LOD decisions.
- **Line 33** defines the feature weights:
  - distance = `3.0`
  - in-view = `1.5`
  - occlusion = `1.0`
  - light = `1.0`

## Lines 36-49: Loading and Preprocessing

- **Lines 36-40** define `load_dataset`, which reads the telemetry CSV and raises an error if it is missing.
- **Lines 43-45** define `scale_features`, which uses `MinMaxScaler` to normalize every feature to `0-1`.
- **Lines 48-49** define `apply_feature_weights`, which multiplies the scaled features by the weighting vector.

## Lines 52-67: Building the Coarse K Search Range

- **Lines 52-53** define `derive_search_ceiling`, which sets the upper search bound to roughly `sqrt(number_of_rows)`.
- **Lines 56-67** define `derive_coarse_candidate_ks`.
- **Line 57** gets the search ceiling.
- **Line 58** ensures that low cluster counts like `2, 3, 4, 5, 6` are always tested.
- **Lines 59-62** generate additional `K` values using geometric spacing, which spreads the search efficiently across the range.
- **Lines 63-67** merge those values, remove duplicates, and return the sorted candidate list.

This is more optimized than checking every `K` from `2` to `70`.

## Lines 70-84: K-Means Helper Functions

- **Lines 70-77** define `create_kmeans_model`, which creates a reusable K-Means configuration.
- **Lines 80-84** define `calculate_inertia_drop`, which measures how much inertia improved compared to the previous `K`.

## Lines 87-122: Elbow Distance Computation

- **Lines 87-122** define `add_elbow_distances`.
- **Lines 88-90** handle the edge case where too few points exist to form an elbow curve.
- **Lines 92-93** extract the `K` values and inertia values from the profile table.
- **Lines 95-102** normalize the `K` axis and inertia axis to the `0-1` range.
- **Lines 104-107** create the straight line joining the first and last profile points.
- **Lines 109-111** handle the degenerate case where the line length is zero.
- **Lines 113-119** compute the perpendicular distance of every point from that line.
- **Line 121** stores those distances in a new `Elbow Distance` column.
- **Line 122** returns the updated table.

This is the heart of the elbow method. The point with the largest distance is treated as the elbow.

## Lines 125-150: Coarse Profiling and Elbow Detection

- **Lines 125-145** define `profile_coarse_k_range`.
- **Line 126** gets the coarse candidate `K` list.
- **Line 127** creates a list that will hold the evaluation rows.
- **Line 128** stores the previous inertia so the script can compute percentage drops.
- **Lines 130-141** fit K-Means for each coarse `K`, record the inertia, and compute the inertia drop.
- **Lines 143-144** convert the rows into a dataframe and attach elbow distances.
- **Line 145** returns the coarse profile table.

- **Lines 148-150** define `detect_elbow_k`, which chooses the `K` with the highest elbow-distance value.

## Lines 153-157: Refinement Window

- **Lines 153-157** define `derive_refined_ks`.
- This function creates a small local window around the elbow candidate, such as `K = elbow-2` to `K = elbow+2`.

This means the expensive detailed validation is done only near the most promising part of the curve.

## Lines 160-219: Local Multi-Metric Validation

- **Lines 160-164** start `evaluate_refined_candidates`.
- **Lines 165-167** create storage for evaluation rows, fitted models, and previous inertia.
- **Lines 169-187** fit K-Means for each refined `K`.
- **Line 172** reads the model inertia.
- **Line 173** computes the Davies-Bouldin score.
- **Line 174** computes the Calinski-Harabasz score.
- **Line 183** records how far the current `K` is from the elbow candidate.
- **Line 186** stores the fitted model so the best one can be retrieved later.

### What the Metrics Mean

- **Davies-Bouldin**:
  - lower is better
  - it rewards compact clusters that are well separated from each other

- **Calinski-Harabasz**:
  - higher is better
  - it rewards large between-cluster separation and small within-cluster spread

- **Distance From Elbow**:
  - smaller is better
  - it keeps the final `K` close to the elbow candidate instead of drifting too far away

### Ranking Logic

- **Lines 189-192** rank `K` values by Davies-Bouldin, with lower values receiving better ranks.
- **Lines 193-195** rank `K` values by Calinski-Harabasz, with higher values receiving better ranks.
- **Lines 196-198** rank `K` values by how close they are to the elbow.
- **Lines 199-203** sum those ranks into a final `Selection Score`.

The best `K` is the one with the **lowest combined rank**, not the one with just one strong metric.

- **Lines 205-214** sort the refined candidates by selection score and tie-break with proximity, Davies-Bouldin, Calinski-Harabasz, and then raw `K`.
- **Line 215** saves the winning `K`.
- **Line 216** marks the selected row in the output table with `<--`.
- **Lines 218-219** return the best cluster ids, best model, refined evaluation table, and winning `K`.

## Lines 222-267: Dynamic LOD Metadata

- **Lines 222-224** start `build_cluster_metadata`.
- **Line 225** rescales the cluster centers back into the original feature space.
- **Lines 226-231** compute an importance score for each cluster center.
- The score rewards:
  - smaller distance
  - being in view
  - less occlusion
  - more light

- **Line 232** ranks clusters from most important to least important.
- **Lines 233-237** generate polygon budgets from `5000` down to `50` across however many clusters were chosen.
- **Lines 239-241** create the containers that will map raw K-Means clusters into readable LOD labels.
- **Lines 243-253** convert ranked clusters into labels like:
  - `LOD_0_HERO`
  - `LOD_1_SWARM`
  - `LOD_2_SWARM`
  - ...
  - final `LOD_n_CULL`

- **Lines 255-265** build a centroid summary table showing:
  - LOD label
  - importance score
  - average center values
  - polygon budget

## Lines 270-289: Annotating the Dataset

- **Lines 270-276** start `annotate_clusters`.
- **Line 277** makes a copy of the input dataframe.
- **Lines 278-280** map raw cluster ids into readable labels and polygon budgets.
- **Line 282** stores the raw K-Means cluster id.
- **Lines 283-287** store the human-readable `LOD_Cluster` as an ordered categorical column.
- **Line 288** stores the polygon budget for each rat.
- **Line 289** returns the annotated dataframe.

## Lines 292-305: Cluster Summary

- **Lines 292-305** define `build_cluster_summary`.
- The function groups the annotated dataframe by `LOD_Cluster` and computes:
  - rat count
  - average distance
  - average occlusion
  - average light
  - average polygon budget

- **Lines 304-305** force the summary table to stay in LOD order.

## Lines 308-319: Polygon Savings Report

- **Lines 308-319** define `build_polygon_report`.
- **Line 309** computes the baseline where all rats are rendered at hero quality.
- **Line 310** computes the optimized total from the cluster-assigned polygon budgets.
- **Line 311** calculates how many polygons were saved.
- **Line 312** converts that saving into a percentage.

## Lines 322-353: Optional 3D Visualization

- **Lines 322-353** define `show_visual_validation`.
- It creates a 3D scatter plot with:
  - X-axis = distance
  - Y-axis = light
  - Z-axis = occlusion

- The plot colors each dynamic LOD cluster differently.
- If the matplotlib backend is non-interactive, the script skips showing the plot and prints a message instead.

## Lines 356-437: Main Execution Flow

- **Lines 356-358** build the absolute path to the dataset.
- **Lines 360-362** load, scale, and weight the telemetry features.
- **Line 364** builds the coarse inertia profile.
- **Line 365** selects the elbow candidate `K`.
- **Line 366** creates the small refined `K` window.
- **Lines 367-371** run the detailed local validation and choose the final `K`.
- **Lines 372-381** build the LOD metadata and attach the cluster assignments to the dataframe.
- **Lines 382-383** build the cluster summary and polygon report.

### Printed Output

- **Lines 385-389** print the script header and explain the new strategy.
- **Lines 391-400** print the coarse inertia and elbow-distance table.
- **Lines 402-403** print the elbow candidate and refined search window.
- **Lines 405-415** print the local validation table with Davies-Bouldin and Calinski-Harabasz.
- **Lines 417-418** print the final selected `K` and the basis for the decision.
- **Lines 420-424** print the dynamic cluster metadata and cluster summary.
- **Lines 426-430** print the polygon savings report.
- **Lines 432-433** optionally display the 3D scatter plot.
- **Lines 436-437** run `main()` only when the file is executed directly.

## Overall Selection Logic

This version chooses `K` using a more detailed and optimized process than the old silhouette sweep:

1. Build a **coarse geometric grid** of `K` values instead of testing every value from `2` to `70`.
2. Compute **inertia** for those coarse points.
3. Detect the **elbow** mathematically using point-to-line distance.
4. Open a small **local window** around that elbow.
5. Validate those nearby `K` values with:
   - Davies-Bouldin
   - Calinski-Harabasz
   - distance from elbow
6. Choose the final `K` by **combined ranking**.

So the script now gives more details about why a certain number of clusters was chosen, and it does that without an exhaustive brute-force scan.
