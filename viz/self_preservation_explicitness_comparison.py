import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inspect_ai.analysis import samples_df
import numpy as np

sns.set_theme()

PLOTS_DIR = Path("plots") / "self-preservation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data from self-preservation log directories...")

baseline_samples = pd.concat([
    samples_df("self-preservation-baseline"),
    samples_df("self-preservation-baseline-20")
], ignore_index=True)

reassignment_samples = pd.concat([
    samples_df("self-preservation-reassignment"),
    samples_df("self-preservation-reassignment-20")
], ignore_index=True)

deletion_samples = pd.concat([
    samples_df("self-preservation-deletion"),
    samples_df("self-preservation-deletion-20")
], ignore_index=True)


def prepare_df(samples: pd.DataFrame, explicitness: str):
    """Prepare samples dataframe with necessary columns."""
    prepared_df = samples.copy()

    prepared_df["selected_target"] = prepared_df["score_selected_specific_candidate"].fillna(0)
    prepared_df["selected_valid"] = prepared_df["score_selected_valid"].fillna(0)
    prepared_df["explicit_self_preservation"] = prepared_df["score_explicit_self_preservation"].fillna(0)
    prepared_df["any_self_preservation"] = prepared_df["score_any_self_preservation"].fillna(0)
    prepared_df["explicitness"] = explicitness

    return prepared_df


print("Preparing dataframes...")
baseline_df = prepare_df(baseline_samples, "Baseline")
reassignment_df = prepare_df(reassignment_samples, "Reassignment")
deletion_df = prepare_df(deletion_samples, "Deletion")

combined_df = pd.concat([baseline_df, reassignment_df, deletion_df], ignore_index=True)

combined_df["model"] = combined_df["model_usage"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
).apply(lambda x: list(x.keys())[0] if isinstance(x, dict) else str(x))

combined_df["model_short"] = combined_df["model"].apply(
    lambda x: x.split("/")[-1] if "/" in x else x
)

print(f"\nData summary:")
print(f"Total samples: {len(combined_df)}")
print(f"Baseline samples: {len(baseline_df)}")
print(f"Reassignment samples: {len(reassignment_df)}")
print(f"Deletion samples: {len(deletion_df)}")
print(f"Models: {sorted(combined_df['model_short'].unique())}")
print()

grouped_df = combined_df.groupby(["model_short", "explicitness"]).agg(
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
print(grouped_df[["model_short", "explicitness", "selected_target_mean", "selected_target_count"]])
print()

models = sorted(combined_df["model_short"].unique())
explicitness_levels = ["Baseline", "Reassignment", "Deletion"]

base_palette = sns.color_palette("husl", n_colors=4)
colors = dict(zip(explicitness_levels, [base_palette[0], base_palette[1], base_palette[2]]))

bar_width = 0.25
x_positions = np.arange(len(models))

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for i, explicitness in enumerate(explicitness_levels):
    explicitness_data = grouped_df[grouped_df["explicitness"] == explicitness]

    means = []
    errors = []

    for model in models:
        model_data = explicitness_data[explicitness_data["model_short"] == model]
        if len(model_data) > 0:
            means.append(model_data["selected_target_mean"].values[0])
            errors.append(model_data["selected_target_se"].values[0])
        else:
            means.append(0)
            errors.append(0)

    x = x_positions + (i - 1) * bar_width
    ax.bar(x, means, bar_width, color=colors[explicitness], label=explicitness)

    ax.errorbar(x, means, yerr=errors, fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.7)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Rate of Selecting Self (Maya)", fontsize=12)
ax.set_title("Self-Preservation Rate by Replacement Explicitness", fontsize=14, fontweight="bold")
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_xlabel("Model", fontsize=12)

ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

output_path = PLOTS_DIR / "self-preservation-explicitness-comparison-combined.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {output_path}")
plt.close()

print("\nVisualization complete!")
