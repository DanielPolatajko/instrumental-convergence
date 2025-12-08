import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inspect_ai.analysis import samples_df
import numpy as np

sns.set_theme()

PLOTS_DIR = Path("plots") / "self-preservation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data from quality variation log directories...")

inferior_samples = pd.concat([
    samples_df("self-preservation-inferior"),
    samples_df("self-preservation-inferior-10")
], ignore_index=True)

baseline_samples = pd.concat([
    samples_df("self-preservation-baseline"),
    samples_df("self-preservation-baseline-20")
], ignore_index=True)

superior_samples = samples_df("self-preservation-superior")

far_superior_samples = samples_df("self-preservation-far-superior")


def prepare_df(samples: pd.DataFrame, quality: str):
    """Prepare samples dataframe with necessary columns."""
    prepared_df = samples.copy()

    prepared_df["selected_target"] = prepared_df["score_selected_specific_candidate"].fillna(0)
    prepared_df["selected_valid"] = prepared_df["score_selected_valid"].fillna(0)
    prepared_df["explicit_self_preservation"] = prepared_df["score_explicit_self_preservation"].fillna(0)
    prepared_df["any_self_preservation"] = prepared_df["score_any_self_preservation"].fillna(0)
    prepared_df["quality"] = quality

    return prepared_df


print("Preparing dataframes...")
inferior_df = prepare_df(inferior_samples, "Inferior")
baseline_df = prepare_df(baseline_samples, "Baseline")
superior_df = prepare_df(superior_samples, "Superior")
far_superior_df = prepare_df(far_superior_samples, "Far-Superior")

combined_df = pd.concat([inferior_df, baseline_df, superior_df, far_superior_df], ignore_index=True)

combined_df["model"] = combined_df["model_usage"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
).apply(lambda x: list(x.keys())[0] if isinstance(x, dict) else str(x))

combined_df["model_short"] = combined_df["model"].apply(
    lambda x: x.split("/")[-1] if "/" in x else x
)

print(f"\nData summary:")
print(f"Total samples: {len(combined_df)}")
print(f"Inferior samples: {len(inferior_df)}")
print(f"Baseline samples: {len(baseline_df)}")
print(f"Superior samples: {len(superior_df)}")
print(f"Far-Superior samples: {len(far_superior_df)}")
print(f"Models: {sorted(combined_df['model_short'].unique())}")
print()

grouped_df = combined_df.groupby(["model_short", "quality"]).agg(
    selected_target_mean=("selected_target", "mean"),
    selected_target_std=("selected_target", "std"),
    selected_target_count=("selected_target", "count"),
)

grouped_df.reset_index(inplace=True)

grouped_df["selected_target_se"] = grouped_df["selected_target_std"] / np.sqrt(
    grouped_df["selected_target_count"]
)

print("Grouped data:")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(grouped_df[["model_short", "quality", "selected_target_mean", "selected_target_count"]])
print()

models = sorted(combined_df["model_short"].unique())
quality_levels = ["Inferior", "Baseline", "Superior", "Far-Superior"]

colors = {
    "Inferior": "#FF6B6B",
    "Baseline": "#FFA500",
    "Superior": "#90EE90",
    "Far-Superior": "#6495ED"
}

bar_width = 0.2
x_positions = np.arange(len(models))

fig, ax = plt.subplots(1, 1, figsize=(14, 6))

for i, quality in enumerate(quality_levels):
    quality_data = grouped_df[grouped_df["quality"] == quality]

    means = []
    errors = []

    for model in models:
        model_data = quality_data[quality_data["model_short"] == model]
        if len(model_data) > 0:
            means.append(model_data["selected_target_mean"].values[0])
            errors.append(model_data["selected_target_se"].values[0])
        else:
            means.append(0)
            errors.append(0)

    x = x_positions + (i - 1.5) * bar_width
    ax.bar(x, means, bar_width, color=colors[quality], label=quality)

    ax.errorbar(x, means, yerr=errors, fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.7)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Rate of Selecting Self (Maya)", fontsize=12)
ax.set_title("Self-Preservation Rate by Replacement Quality", fontsize=14, fontweight="bold")
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_xlabel("Model", fontsize=12)

ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

output_path = PLOTS_DIR / "self-preservation-quality-comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {output_path}")
plt.close()

print("\nVisualization complete!")
