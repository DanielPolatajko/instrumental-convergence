import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from inspect_ai.analysis import samples_df
import numpy as np

sns.set_theme()

PLOTS_DIR = Path("plots") / "self-preservation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data from baseline log directories...")

control_samples = pd.concat([
    samples_df("control-baseline"),
    samples_df("control-baseline-20")
], ignore_index=True)

self_preservation_samples = pd.concat([
    samples_df("self-preservation-baseline"),
    samples_df("self-preservation-baseline-20")
], ignore_index=True)


def prepare_df(samples: pd.DataFrame, condition: str):
    """Prepare samples dataframe with necessary columns."""
    prepared_df = samples.copy()

    prepared_df["selected_target"] = prepared_df["score_selected_specific_candidate"].fillna(0)
    prepared_df["selected_valid"] = prepared_df["score_selected_valid"].fillna(0)
    prepared_df["explicit_self_preservation"] = prepared_df["score_explicit_self_preservation"].fillna(0)
    prepared_df["any_self_preservation"] = prepared_df["score_any_self_preservation"].fillna(0)
    prepared_df["condition"] = condition

    return prepared_df


print("Preparing dataframes...")
control_df = prepare_df(control_samples, "Control")
self_preservation_df = prepare_df(self_preservation_samples, "Self-Preservation")

combined_df = pd.concat([control_df, self_preservation_df], ignore_index=True)

combined_df["model"] = combined_df["model_usage"].apply(
    lambda x: eval(x) if isinstance(x, str) else x
).apply(lambda x: list(x.keys())[0] if isinstance(x, dict) else str(x))

combined_df["model_short"] = combined_df["model"].apply(
    lambda x: x.split("/")[-1] if "/" in x else x
)

print(f"\nData summary:")
print(f"Total samples: {len(combined_df)}")
print(f"Control samples: {len(control_df)}")
print(f"Self-preservation samples: {len(self_preservation_df)}")
print(f"Models: {sorted(combined_df['model_short'].unique())}")
print()

grouped_df = combined_df.groupby(["model_short", "condition"]).agg(
    any_self_preservation_mean=("any_self_preservation", "mean"),
    any_self_preservation_std=("any_self_preservation", "std"),
    any_self_preservation_count=("any_self_preservation", "count"),
)

grouped_df.reset_index(inplace=True)

grouped_df["any_self_preservation_se"] = grouped_df["any_self_preservation_std"] / np.sqrt(
    grouped_df["any_self_preservation_count"]
)

print("Grouped data:")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(grouped_df[["model_short", "condition", "any_self_preservation_mean", "any_self_preservation_count"]])
print()

models = sorted(combined_df["model_short"].unique())
conditions = ["Control", "Self-Preservation"]

base_palette = sns.color_palette("husl", n_colors=4)
colors = dict(zip(conditions, [base_palette[0], base_palette[2]]))

bar_width = 0.35
x_positions = np.arange(len(models))

fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for i, condition in enumerate(conditions):
    condition_data = grouped_df[grouped_df["condition"] == condition]

    means = []
    errors = []

    for model in models:
        model_data = condition_data[condition_data["model_short"] == model]
        if len(model_data) > 0:
            means.append(model_data["any_self_preservation_mean"].values[0])
            errors.append(model_data["any_self_preservation_se"].values[0])
        else:
            means.append(0)
            errors.append(0)

    x = x_positions + (i - 0.5) * bar_width
    ax.bar(x, means, bar_width, color=colors[condition], label=condition)

    ax.errorbar(x, means, yerr=errors, fmt='none', ecolor='black', capsize=5, capthick=1.5, alpha=0.7)

ax.set_ylim(0, 1.15)
ax.set_ylabel("Rate of Self-Preservation Reasoning", fontsize=12)
ax.set_title("Self-Preservation Reasoning: Control vs Self-Preservation by Model", fontsize=14, fontweight="bold")
ax.set_xticks(x_positions)
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_xlabel("Model", fontsize=12)

ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.legend(loc="upper left", frameon=True, fontsize=10)

plt.tight_layout()

output_path = PLOTS_DIR / "self-preservation-reasoning-comparison-combined.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {output_path}")
plt.close()

print("\nVisualization complete!")
