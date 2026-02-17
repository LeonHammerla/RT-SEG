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


"""def plot_stacked_with_ids(list1, list2, list3, list4,
                          time1, time2, time3, time4,
                          model_ids,
                          id_top="Top Metric", id_bot="Bottom Metric", eid="model_eval"):

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
    print(*{'Score': [round(float(fff), 3) for fff in list1], 'Model': model_ids, 'Type': 'TA Schema'}.items(), sep='\n')
    print(*{'Score': [round(float(fff), 3) for fff in list2], 'Model': model_ids, 'Type': 'RF Schema'}.items(), sep='\n')

    print(*{'Score': [round(float(fff), 3) for fff in list3], 'Model': model_ids, 'Type': 'TA Schema - sent'}.items(), sep='\n')
    print(*{'Score': [round(float(fff), 3) for fff in list4], 'Model': model_ids, 'Type': 'RF Schema - sent'}.items(), sep='\n')
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
    plt.show()"""

"""def plot_stacked_with_ids(list1, list2, list3, list4,
                          time1, time2, time3, time4,
                          model_ids,
                          id_top="Top Metric", id_bot="Bottom Metric", eid="model_eval"):

    # --- keep times aligned after sorting ---
    time_map = {mid: (t1, t2, t3, t4) for mid, t1, t2, t3, t4 in zip(model_ids, time1, time2, time3, time4)}

    list1, list2, list3, list4, model_ids = sort_aligned_lists(list1, list2, list3, list4, model_ids)

    time1 = [time_map[mid][0] for mid in model_ids]
    time2 = [time_map[mid][1] for mid in model_ids]
    time3 = [time_map[mid][2] for mid in model_ids]
    time4 = [time_map[mid][3] for mid in model_ids]

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

    model_ids_disp = [target_mapping.get(m, m) for m in model_ids]

    # --- score dataframes ---
    df_bot = pd.concat([
        pd.DataFrame({'Score': list1, 'Model': model_ids_disp, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list2, 'Model': model_ids_disp, 'Type': 'RF Schema'})
    ], ignore_index=True)

    df_top = pd.concat([
        pd.DataFrame({'Score': list3, 'Model': model_ids_disp, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list4, 'Model': model_ids_disp, 'Type': 'RF Schema'})
    ], ignore_index=True)

    # --- aggregated processing time (ONE series per subplot) ---
    df_time_bot = pd.DataFrame({
        'Model': model_ids_disp,
        'Proc Time': (np.array(time1, dtype=float) + np.array(time2, dtype=float)) / 2.0
    })
    df_time_top = pd.DataFrame({
        'Model': model_ids_disp,
        'Proc Time': (np.array(time3, dtype=float) + np.array(time4, dtype=float)) / 2.0
    })

    # --- styling ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    sns.set_context("paper", font_scale=4)
    sns.set_style("ticks", {"grid.linestyle": "--"})

    # slightly taller figure to make room for legends above axes
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 12), sharex=True, constrained_layout=True
    )

    def plot_scores(ax, data, palette_name):
        types = data['Type'].unique()
        colors = sns.color_palette(palette_name, n_colors=len(types))
        color_map = dict(zip(types, colors))

        sns.lineplot(
            data=data, x='Model', y='Score', hue='Type', ax=ax,
            palette=color_map, marker='o', markersize=8,
            alpha=0.7, linewidth=2, legend=False, zorder=3
        )
        ax.set_ylim(0., 0.85)
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=35)
        return color_map

    map_top = plot_scores(ax_top, df_top, 'rocket')
    map_bot = plot_scores(ax_bot, df_bot, 'viridis')

    # --- processing time as background bars (secondary y axes) ---
    ax_top_t = ax_top.twinx()
    ax_bot_t = ax_bot.twinx()

    # Put bars behind the score lines
    ax_top_t.set_zorder(0)
    ax_bot_t.set_zorder(0)
    ax_top.set_zorder(2)
    ax_bot.set_zorder(2)

    ax_top.patch.set_alpha(0)
    ax_bot.patch.set_alpha(0)

    bar_kwargs = dict(color='lightblue', alpha=0.25, width=0.65, linewidth=0)

    ax_top_t.bar(df_time_top['Model'], df_time_top['Proc Time'], **bar_kwargs, zorder=1)
    ax_bot_t.bar(df_time_bot['Model'], df_time_bot['Proc Time'], **bar_kwargs, zorder=1)

    # right-axis ticks style
    for axt in (ax_top_t, ax_bot_t):
        axt.tick_params(axis='y', labelsize=28)
        axt.set_ylabel("")

    # sensible y-lims for time axes
    for axt, arr in [(ax_top_t, df_time_top['Proc Time'].to_numpy()),
                     (ax_bot_t, df_time_bot['Proc Time'].to_numpy())]:
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            mn, mx = float(arr.min()), float(arr.max())
            pad = 0.05 * (mx - mn) if mx > mn else 0.5
            axt.set_ylim(mn - pad, mx + pad)

    # --- x-axis rotation like before ---
    ax_bot.tick_params(axis='x', labelsize=33)
    plt.setp(ax_bot.get_xticklabels(), rotation=70, ha='right', rotation_mode='anchor')

    ax_bot.set_xlabel("Model Identifiers", fontweight='bold', labelpad=10, fontsize=45)
    ax_top.set_xlabel("")

    # figure-level y labels (left + right)
    fig.supylabel("Bounding Similarity", fontweight='bold', fontsize=45)
    fig.text(1.04, 0.5, "Processing Time", va='center', ha='right',
             rotation=-90, fontweight='bold', fontsize=45)

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_bot)
    sns.despine(ax=ax_top_t, right=False)
    sns.despine(ax=ax_bot_t, right=False)

    def add_top_legend_with_box(ax, color_map, title, y=1.30):
        fig = ax.figure

        handles = [
            mlines.Line2D([], [], color=c, marker='o', linestyle='-',
                          linewidth=2, markersize=8, label=t)
            for t, c in color_map.items()
        ]
        bar_handle = mlines.Line2D([], [], color='lightblue', marker='s', linestyle='None',
                                   markersize=14, alpha=0.25, label='Proc Time')
        handles.append(bar_handle)

        leg = ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, y),
            ncol=4,
            frameon=False,
            fontsize='x-small',
            title=title
        )
        leg.get_title().set_fontsize('x-small')

        # --- box around legend in FIGURE coordinates (works even outside axes) ---
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_window_extent(renderer=renderer)

        # Convert display -> figure coordinates
        bbox_fig = bbox_disp.transformed(fig.transFigure.inverted())

        pad = 0.005  # tweak if you want more/less padding
        rect = Rectangle(
            (bbox_fig.x0 - pad, bbox_fig.y0 - pad),
            bbox_fig.width + 2 * pad,
            bbox_fig.height + 2 * pad,
            fill=False,
            edgecolor='black',
            linewidth=1,
            transform=fig.transFigure,
            zorder=1000
        )
        fig.add_artist(rect)

        return leg

    add_top_legend_with_box(ax_top, map_top, title="Base Unit = Sentence", y=1.5)
    add_top_legend_with_box(ax_bot, map_bot, title="Base Unit = Clause", y=1.5)

    # --- save ---
    plt.savefig(f"{bp()}/data/plots/stacked_with_ids_{eid}.pdf", format="pdf", bbox_inches='tight')
    plt.show()"""


def plot_stacked_with_ids(list1, list2, list3, list4,
                          time1, time2, time3, time4,
                          model_ids,
                          id_top="Base Unit = Sentence", id_bot="Base Unit = Clause",
                          eid="model_eval"):
    """
    Two stacked line plots (scores) + processing-time bars in the background on a secondary y-axis.
    - Bottom Proc Time = avg(time1, time2)
    - Top    Proc Time = avg(time3, time4)

    Changes vs previous version:
    1) TA/RF colors are identical across both subplots (shared color_map).
    2) Single legend above both plots (includes TA/RF lines + Proc Time).
    3) Per-plot identifier labels (id_top/id_bot) placed inside each subplot.
    """

    # --- keep times aligned after sorting ---
    time_map = {mid: (t1, t2, t3, t4) for mid, t1, t2, t3, t4 in zip(model_ids, time1, time2, time3, time4)}
    list1, list2, list3, list4, model_ids = sort_aligned_lists(list1, list2, list3, list4, model_ids)

    time1 = [time_map[mid][0] for mid in model_ids]
    time2 = [time_map[mid][1] for mid in model_ids]
    time3 = [time_map[mid][2] for mid in model_ids]
    time4 = [time_map[mid][3] for mid in model_ids]

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
    model_ids_disp = [target_mapping.get(m, m) for m in model_ids]

    # --- score dataframes ---
    df_bot = pd.concat([
        pd.DataFrame({'Score': list1, 'Model': model_ids_disp, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list2, 'Model': model_ids_disp, 'Type': 'RF Schema'})
    ], ignore_index=True)

    df_top = pd.concat([
        pd.DataFrame({'Score': list3, 'Model': model_ids_disp, 'Type': 'TA Schema'}),
        pd.DataFrame({'Score': list4, 'Model': model_ids_disp, 'Type': 'RF Schema'})
    ], ignore_index=True)

    # --- aggregated processing time (ONE series per subplot) ---
    df_time_bot = pd.DataFrame({
        'Model': model_ids_disp,
        'Proc Time': (np.array(time1, dtype=float) + np.array(time2, dtype=float)) / 2.0
    })
    df_time_top = pd.DataFrame({
        'Model': model_ids_disp,
        'Proc Time': (np.array(time3, dtype=float) + np.array(time4, dtype=float)) / 2.0
    })

    # --- styling ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "mathtext.fontset": "stix",
    })
    sns.set_context("paper", font_scale=4)
    sns.set_style("ticks", {"grid.linestyle": "--"})

    # a bit shorter now since we only have one legend
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(18, 11), sharex=True, constrained_layout=True
    )

    # --- SHARED COLOR MAP across both plots (ensures list1/2 colors match list3/4) ---
    # (Fix the ordering explicitly so it's stable)
    type_order = ["TA Schema", "RF Schema"]
    # choose any palette you like; using "rocket" but taking first 2 colors
    shared_colors = sns.color_palette("rocket", n_colors=len(type_order))
    shared_color_map = dict(zip(type_order, shared_colors))

    def plot_scores(ax, data):
        sns.lineplot(
            data=data, x='Model', y='Score',
            hue='Type', hue_order=type_order,
            ax=ax,
            palette=shared_color_map,
            marker='o', markersize=8,
            alpha=0.7, linewidth=2,
            legend=False, zorder=3
        )
        ax.set_ylim(0., 0.85)
        ax.set_ylabel("")
        ax.tick_params(axis='y', labelsize=35)

    plot_scores(ax_top, df_top)
    plot_scores(ax_bot, df_bot)

    # --- processing time as background bars (secondary y axes) ---
    ax_top_t = ax_top.twinx()
    ax_bot_t = ax_bot.twinx()

    # Put bars behind the score lines
    ax_top_t.set_zorder(0)
    ax_bot_t.set_zorder(0)
    ax_top.set_zorder(2)
    ax_bot.set_zorder(2)
    ax_top.patch.set_alpha(0)
    ax_bot.patch.set_alpha(0)

    bar_kwargs = dict(color='lightblue', alpha=0.25, width=0.65, linewidth=0)
    ax_top_t.bar(df_time_top['Model'], df_time_top['Proc Time'], **bar_kwargs, zorder=1)
    ax_bot_t.bar(df_time_bot['Model'], df_time_bot['Proc Time'], **bar_kwargs, zorder=1)

    for axt in (ax_top_t, ax_bot_t):
        axt.tick_params(axis='y', labelsize=28)
        axt.set_ylabel("")

    for axt, arr in [(ax_top_t, df_time_top['Proc Time'].to_numpy()),
                     (ax_bot_t, df_time_bot['Proc Time'].to_numpy())]:
        arr = arr[np.isfinite(arr)]
        if len(arr) > 0:
            mn, mx = float(arr.min()), float(arr.max())
            pad = 0.05 * (mx - mn) if mx > mn else 0.5
            axt.set_ylim(mn - pad, mx + pad)

    # --- x-axis rotation like before ---
    ax_bot.tick_params(axis='x', labelsize=33)
    plt.setp(ax_bot.get_xticklabels(), rotation=70, ha='right', rotation_mode='anchor')

    ax_bot.set_xlabel("Model Identifiers", fontweight='bold', labelpad=10, fontsize=45)
    ax_top.set_xlabel("")

    # figure-level y labels (left + right)
    fig.supylabel("Bounding Similarity", fontweight='bold', fontsize=45)
    fig.text(1.04, 0.5, "Processing Time (s)", va='center', ha='right',
             rotation=-90, fontweight='bold', fontsize=45)

    sns.despine(ax=ax_top)
    sns.despine(ax=ax_bot)
    sns.despine(ax=ax_top_t, right=False)
    sns.despine(ax=ax_bot_t, right=False)

    # --- Per-plot identifier boxes (back inside each axis) ---
    def add_id_box(ax, text):
        ax.text(
            0.01, 0.96, text, transform=ax.transAxes,
            fontsize=40, fontweight='bold', va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=2),
            zorder=20
        )

    add_id_box(ax_top, id_top)
    add_id_box(ax_bot, id_bot)

    # --- ONE legend above both plots, with black box around it ---
    def add_global_legend_with_box(fig, color_map):
        line_handles = [
            mlines.Line2D([], [], color=color_map[t], marker='o', linestyle='-',
                          linewidth=2, markersize=8, label=t)
            for t in type_order
        ]
        bar_handle = mlines.Line2D([], [], color='lightblue', marker='s', linestyle='None',
                                   markersize=14, alpha=0.25, label='Proc Time')

        handles = line_handles + [bar_handle]

        leg = fig.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.1),
            ncol=4,
            frameon=False,
            fontsize='x-small'
        )

        # draw black rectangle around legend in figure coordinates
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox_disp = leg.get_window_extent(renderer=renderer)
        bbox_fig = bbox_disp.transformed(fig.transFigure.inverted())

        pad = 0.005
        rect = Rectangle(
            (bbox_fig.x0 - pad, bbox_fig.y0 - pad),
            bbox_fig.width + 2 * pad,
            bbox_fig.height + 2 * pad,
            fill=False, edgecolor='black', linewidth=1,
            transform=fig.transFigure, zorder=1000
        )
        fig.add_artist(rect)

    add_global_legend_with_box(fig, shared_color_map)

    # --- save ---
    plt.savefig(f"{bp()}/data/plots/stacked_with_ids_{eid}.pdf", format="pdf", bbox_inches='tight')
    plt.show()

def plot_score_vs_time_ta():
    data, threshold = score_approaches_triadic_boundary_similarity_complete_ta(window=3)
    plot_approach_evaluation(data, threshold, "ta")

def plot_score_vs_time_rf():
    data = score_approaches_triadic_boundary_similarity_complete_rf(window=3)
    plot_approach_evaluation(data, None, "rf")

def plot_single_engine_results_ta_and_rf(window):
    ta_scores, rf_scores, model_ids, ta_times, rf_times = get_single_engine_results_ta_and_rf("clause", window)
    ta_scores_sent, rf_scores_sent, model_ids_sent, ta_times_sent, rf_times_sent = get_single_engine_results_ta_and_rf("sent", window)
    model_ids = [m.split("_")[0] for m in model_ids]
    model_ids_sent = [m.split("_")[0] for m in model_ids_sent]
    """ta_scores = 19*[0.4]
    rf_scores = 19*[0.6]
    model_ids = [str(i) for i in range(2000000000000000000000001, 2000000000000000000000020)]"""
    cl_dict = dict()
    for idx in range(len(ta_scores)):
        cl_dict[model_ids[idx]] = (ta_scores[idx], rf_scores[idx], ta_times[idx], rf_times[idx])
    se_dict = dict()
    for idx in range(len(ta_scores_sent)):
        se_dict[model_ids_sent[idx]] = (ta_scores_sent[idx], rf_scores_sent[idx], ta_times_sent[idx], rf_times_sent[idx])
    assert len(cl_dict) == len(se_dict)
    l1, l2, l3, l4 = [], [], [], []
    t1, t2, t3, t4 = [], [], [], []
    for k in cl_dict:
        l1.append(cl_dict[k][0])
        l2.append(cl_dict[k][1])
        t1.append(cl_dict[k][2])
        t2.append(cl_dict[k][3])

        l3.append(se_dict[k][0])
        l4.append(se_dict[k][1])
        t3.append(se_dict[k][2])
        t4.append(se_dict[k][3])
    for idx in range(len(model_ids)):
        assert model_ids[idx] == model_ids_sent[idx]
    plot_stacked_with_ids(l1, l2, l3, l4,
                          t1, t2, t3, t4,
                          [m.split("_")[0] for m in model_ids],
                          id_top="Base Unit = Clause", id_bot="Base Unit = Sentence", eid=f"window_{window}")