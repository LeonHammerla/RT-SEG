import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import ticker
import matplotlib.lines as mlines
from matplotlib.patches import FancyBboxPatch, Rectangle

from .seg_utils import bp
from .seg_eval_utils import score_approaches_triadic_boundary_similarity_complete_ta, \
    score_approaches_triadic_boundary_similarity_complete_rf, get_single_engine_results_ta_and_rf


def plot_approach_evaluation(data_dict, threshold, eid: str):
    df = pd.DataFrame(data_dict)

    # --- 1. PRE-DEFINE MAPPINGS (Ensure legend and plot match) ---
    unique_labels = sorted(df['labels'].unique())
    colors = sns.color_palette('viridis', n_colors=len(unique_labels))
    label_color_map = dict(zip(unique_labels, colors))

    unique_shapes = sorted(df['shapes'].unique())
    marker_list = ['o', "*", 'P', 'X', '^', 's', 'd']
    shape_marker_map = dict(zip(unique_shapes, marker_list[:len(unique_shapes)]))

    # Setup Styling
    # 1. Update the rcParams to use STIX fonts
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    sns.set_context("paper", font_scale=2.3)
    sns.set_style("ticks", {"grid.linestyle": "--"})
    if threshold is None:
        fig, ax = plt.subplots(figsize=(10, 8.6))
    else:
        fig, ax = plt.subplots(figsize=(10, 7.5))

    # 2. Warp functions
    def forward(x): return np.power(x, 0.25)

    def inverse(x): return np.power(x, 4)

    # 3. Plotting with EXPLICIT maps
    sns.scatterplot(
        data=df, x='time', y='score',
        hue='labels', palette=label_color_map,  # Use the map
        style='shapes', markers=shape_marker_map,  # Use the map
        s=120, alpha=0.9,
        edgecolor='w', linewidth=0.5, zorder=3,
        legend=False
    )

    # 4. Axes & Scale (Keeping your existing logic)
    plt.xscale('function', functions=(forward, inverse))
    ticks = [0, 1, 15, 80, 250, 600, 1000]
    plt.xticks(ticks)
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.xlim(0, df['time'].max() * 1.05)
    plt.ylim(0.2, 1.05)

    # 5. Threshold Line
    if threshold is not None:
        plt.axhline(y=threshold, color='#d62728', linestyle='--', linewidth=1.5, zorder=2)

    # 6. Labels
    plt.xlabel(r"Time Complexity (s) [$\mathbf{\sqrt[4]{t}}$ scale]", fontweight='bold')
    plt.ylabel('Bounding Similarity', fontweight='bold')
    sns.despine()

    # ---------------------------------------------------------
    # 7. MANUAL LEGEND CONSTRUCTION (Now using the same maps)
    # ---------------------------------------------------------
    hue_handles = [mlines.Line2D([], [], color=label_color_map[lab], marker='o',
                                 linestyle='None', markersize=8, label=str(lab))
                   for lab in unique_labels]

    shape_handles = [mlines.Line2D([], [], color='gray', marker=shape_marker_map[shp],
                                   linestyle='None', markersize=8, label=str(shp))
                     for shp in unique_shapes]

    if threshold is not None:
        thresh_handle = [mlines.Line2D([], [], color='#d62728', linestyle='--',
                                       linewidth=1.5, label=f'Human ({threshold:.2f})')]

    if threshold is None:
        offs = 0.1
        offs2 = 0.013
    else:
        offs = 0
        offs2 = 0
    # Create Legends
    leg1 = ax.legend(handles=hue_handles, title="Engine Comb.", loc='lower left',
                     bbox_to_anchor=(0.05, 0.7+offs), ncol=3, frameon=False, fontsize='small',
                     borderpad=0.2,  # Increase for more "breathing room" inside the box
                     handletextpad=0.2,  # Decrease to pull text closer to the markers
                     columnspacing=0.8,
                     labelspacing=0.2
                     )
    ax.add_artist(leg1)

    leg2 = ax.legend(handles=shape_handles, title="Late Fusion", loc='lower center',
                     bbox_to_anchor=(0.75, 0.82+offs-offs2), ncol=2, frameon=False, fontsize='small',
                     borderpad=0.2,  # Increase for more "breathing room" inside the box
                     handletextpad=0.1,  # Decrease to pull text closer to the markers
                     columnspacing=0.8,
                     labelspacing=0.1
                     )
    ax.add_artist(leg2)

    if threshold is not None:
        # Using loc='lower right' for threshold as requested
        leg3 = ax.legend(handles=thresh_handle, title="Baseline", loc='lower center',
                         bbox_to_anchor=(0.75, 0.7), frameon=False, fontsize='small',
                         borderpad=0.2,  # Increase for more "breathing room" inside the box
                         handletextpad=0.1,  # Decrease to pull text closer to the markers
                         columnspacing=0.8,
                         labelspacing=0.1
                         )

    # --- 8. WRAPPER BOX (Refined) ---
    plt.tight_layout()
    fig.canvas.draw()

    inv = ax.transAxes.inverted()
    b1 = inv.transform_bbox(leg1.get_window_extent())
    b2 = inv.transform_bbox(leg2.get_window_extent())

    x0, y0 = min(b1.x0, b2.x0), min(b1.y0, b2.y0)
    x1, y1 = max(b1.x1, b2.x1), max(b1.y1, b2.y1)

    pad = 0.015
    outer_box = Rectangle(
        (x0 - pad, y0 - pad),
        (x1 - x0) + (2 * pad), (y1 - y0) + (2 * pad),
        fill=False, edgecolor='black', linewidth=1.5,
        transform=ax.transAxes, clip_on=False, zorder=0
    )
    ax.add_patch(outer_box)

    plt.tight_layout()
    plt.savefig(f"{bp()}/data/plots/score_vs_time_{eid}.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def sort_aligned_lists(key_list, *other_lists):
    """
    Sorts multiple lists based on the order of key_list.

    Args:
        key_list: The list used for sorting criteria.
        *other_lists: Any number of additional lists to be reordered.

    Returns:
        A tuple of sorted lists.
    """
    # Combine all lists into a single iterable of tuples
    combined = zip(key_list, *other_lists)

    # Sort based on the first element (the key_list values)
    # Use reverse=True inside sorted() if you want descending order
    sorted_combined = sorted(combined, key=lambda x: x[0])

    # Unzip the sorted tuples back into individual lists
    unzipped = list(zip(*sorted_combined))

    # Convert tuples back to lists and return
    return tuple(list(item) for item in unzipped)


def plot_stacked_with_ids(list1, list2, list3, list4, model_ids,
                          id_top="Top Metric", id_bot="Bottom Metric", eid="model_eval"):
    """
    Plots stacked lines with internal text IDs to differentiate plots
    without adding external margins.
    """
    list1, list2, list3, list4, model_ids = sort_aligned_lists(list1, list2, list3, list4, model_ids)
    target_mapping = {
    "RTRuleRegex": "Regex",
    "RTNewLine": "Newline",
    "RTLLMForcedDecoderBased": "Forced-Decod.",
    "RTLLMSurprisal": "Surprisal",
    "RTLLMEntropy": "Entropy",
    "RTLLMTopKShift": "Top-K Shift",
    "RTLLMFlatnessBreak": "Flatness Break",
    "RTBERTopicSegmentation": "BERTopic",
    "RTEmbeddingBasedSemanticShift": "SemShift",
    "RTEntailmentBasedSegmentation": "Entailment",
    "RTZeroShotSeqClassification": "ZeroShot",
    "RTZeroShotSeqClassificationTA": "ZeroShot (TA)",
    "RTZeroShotSeqClassificationRF": "ZeroShot (RF)",
    "RTPRMBase": "PRM Base",
    "RTLLMThoughtAnchor": "LLM (TA)",
    "RTLLMReasoningFlow": "LLM (RF)",
    "RTLLMArgument": "LLM Arg.",
    "RTLLMOffsetBased": "LLM Offset",
    "RTLLMSegUnitBased": "LLM Unit"
}
    model_ids = [target_mapping[m] for m in model_ids]
    # --- 1. DATA PREPARATION ---
    df_bot = pd.concat([
        pd.DataFrame({'Score': list1, 'Model': model_ids, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list2, 'Model': model_ids, 'Type': 'RF Schema'})
    ])
    # Top Plot Data
    df_top = pd.concat([
        pd.DataFrame({'Score': list3, 'Model': model_ids, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list4, 'Model': model_ids, 'Type': 'RF Schema'})
    ])

    # --- 2. STYLING SETUP ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    sns.set_context("paper", font_scale=4)
    sns.set_style("ticks", {"grid.linestyle": "--"})

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                         constrained_layout=True)

    def plot_on_axis(ax, data, palette_name, plot_id):
        types = data['Type'].unique()
        colors = sns.color_palette(palette_name, n_colors=len(types))
        color_map = dict(zip(types, colors))

        sns.lineplot(
            data=data, x='Model', y='Score', hue='Type', ax=ax,
            palette=color_map, marker='o', markersize=8,
            alpha=0.7, linewidth=2, legend=False, zorder=3
        )
        ax.set_ylim(0., 0.85)
        ax.set_ylabel("", fontweight='bold')
        ax.tick_params(axis='y', labelsize=35)

        # --- ADDING THE ID TEXT INSIDE THE PLOT ---
        # This places text in the top-left (0.02, 0.95) relative to the axis
        ax.text(0.01, 0.96, plot_id, transform=ax.transAxes,
                fontsize=45, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=2))

        return color_map

    # Plot and add the internal IDs
    map_top = plot_on_axis(ax_top, df_top, 'rocket', id_top)
    map_bot = plot_on_axis(ax_bot, df_bot, 'viridis', id_bot)

    # --- 3. AXES & ROTATION ---
    plt.xticks(rotation=70, ha='right', fontsize=33)

    ax_bot.set_xlabel("Model Identifiers", fontweight='bold', labelpad=10, fontsize=45)
    ax_top.set_xlabel("")

    # Global Y-Label centered on the figure
    fig.supylabel("Bounding Similarity", fontweight='bold',
                  fontsize=45
                  )

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_bot)

    # --- 4. LEGENDS & BOXES ---
    def add_boxed_legend(ax, color_map):
        handles = [
            mlines.Line2D([], [], color=c, marker='o', linestyle='-',
                          linewidth=2, markersize=8, label=t)
            for t, c in color_map.items()
        ]
        leg = ax.legend(handles=handles, loc='lower right', frameon=False, fontsize='x-small')

        fig.canvas.draw()
        inv = ax.transAxes.inverted()
        bbox = inv.transform_bbox(leg.get_window_extent())
        pad = 0.01
        ax.add_patch(Rectangle(
            (bbox.x0 - pad, bbox.y0 - pad), bbox.width + (2 * pad), bbox.height + (2 * pad),
            fill=False, edgecolor='black', linewidth=1, transform=ax.transAxes, zorder=5
        ))

    add_boxed_legend(ax_top, map_top)
    add_boxed_legend(ax_bot, map_bot)

    # --- 5. SAVE ---
    plt.savefig(f"{bp()}/data/plots/stacked_with_ids_{eid}.pdf", format="pdf", bbox_inches='tight')
    plt.show()


def plot_score_vs_time_ta():
    data, threshold = score_approaches_triadic_boundary_similarity_complete_ta(window=3)
    plot_approach_evaluation(data, threshold, "ta")

def plot_score_vs_time_rf():
    data = score_approaches_triadic_boundary_similarity_complete_rf(window=3)
    plot_approach_evaluation(data, None, "rf")

def plot_single_engine_results_ta_and_rf(window):
    ta_scores, rf_scores, model_ids = get_single_engine_results_ta_and_rf("clause", window)
    ta_scores_sent, rf_scores_sent, model_ids_sent = get_single_engine_results_ta_and_rf("sent", window)
    model_ids = [m.split("_")[0] for m in model_ids]
    model_ids_sent = [m.split("_")[0] for m in model_ids_sent]
    """ta_scores = 19*[0.4]
    rf_scores = 19*[0.6]
    model_ids = [str(i) for i in range(2000000000000000000000001, 2000000000000000000000020)]"""
    cl_dict = dict()
    for idx in range(len(ta_scores)):
        cl_dict[model_ids[idx]] = (ta_scores[idx], rf_scores[idx])
    se_dict = dict()
    for idx in range(len(ta_scores_sent)):
        se_dict[model_ids_sent[idx]] = (ta_scores_sent[idx], rf_scores_sent[idx])
    assert len(cl_dict) == len(se_dict)
    l1, l2, l3, l4 = [], [], [], []
    for k in cl_dict:
        l1.append(cl_dict[k][0])
        l2.append(cl_dict[k][1])
        l3.append(se_dict[k][0])
        l4.append(se_dict[k][1])
    for idx in range(len(model_ids)):
        assert model_ids[idx] == model_ids_sent[idx]
    plot_stacked_with_ids(l1, l2, l3, l4, [m.split("_")[0] for m in model_ids],
                          id_top="Base Unit = Clause", id_bot="Base Unit = Sentence", eid=f"window_{window}")