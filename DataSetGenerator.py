from pathlib import Path # used to handle file paths

import numpy as np #used here to generate random numbers for rat attributes.
import pandas as pd #creates the structure (DataFrame) and saves it as a CSV



ROW_COUNT = 5000
OUTPUT_FILE = "rat_swarm_telemetry.csv"


def generate_rat_swarm_dataset(row_count: int = ROW_COUNT) -> pd.DataFrame:
    """Generate synthetic telemetry for rat entities in the Plague-Swarm-LOD project."""
    rng = np.random.default_rng(seed=42) #just random seed

    dataset = pd.DataFrame(
        {
            # Distance from the player (0.0: at feet, 100.0: horizon). 
            # Primary factor for mesh simplification and LOD tiering.
            "distance_to_player": rng.uniform(0.0, 100.0, row_count),

            # Visibility check (1: inside camera view, 0: off-screen). 
            # Weighted at 70% in-view to simulate a dense forward-facing swarm.
            "in_view_frustum": rng.choice([0, 1], size=row_count, p=[0.3, 0.7]),

            # Percentage of object hiding the entity (0.0: clear, 1.0: fully hidden).
            # Helps the model identify rats that can be culled even if they are close.
            "occlusion_factor": rng.uniform(0.0, 1.0, row_count),

            # Environmental brightness (0.0: pitch black, 1.0: bright torchlight).
            # Determines if texture detail can be aggressively downscaled in shadows.
            "light_intensity": rng.uniform(0.0, 1.0, row_count),
        }
    )

    return dataset


def main() -> None:
    output_path = Path(__file__).resolve().parent / OUTPUT_FILE
    dataset = generate_rat_swarm_dataset()
    dataset.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(dataset.head())


if __name__ == "__main__":
    main()
