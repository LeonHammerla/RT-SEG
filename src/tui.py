import random
from typing import List, Tuple, Dict, Literal
import colorsys
from rich.text import Text
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on, events
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, Select, SelectionList

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSentBased,
                             RTRuleRegex,
                             RTNewLine,
                             RTBERTopicSegmentation,
                             RTZeroShotSeqClassification,
                             bp,
                             sdb_login,
                             load_prompt,
                             load_example_trace,
                             RTLLMSurprisal,
                             RTLLMEntropy,
                             RTLLMTopKShift,
                             RTLLMFlatnessBreak,
                             export_gold_set,
                             RTEmbeddingBasedSemanticShift,
                             RTPRMBase,
                             RTEntailmentBasedSegmentation,
                             RTSeg,
                             OffsetFusionFuzzy,
                             OffsetFusionGraph,
                             OffsetFusionMerge,
                             OffsetFusionVoting,
                             OffsetFusionFlatten,
                             OffsetFusionIntersect,
                             OffsetFusion,
                             LabelFusion
                             )


def generate_label_colors(labels: list[str]) -> dict[str, str]:
    """
    Generates a unique, high-contrast hex color for each label
    by spreading hues evenly around the color wheel.
    """
    color_map = {}
    n = len(labels)

    for i, label in enumerate(labels):
        # Calculate hue: spread evenly from 0.0 to 1.0
        hue = i / n

        # We keep Saturation (0.7) and Lightness (0.6) constant
        # to ensure the colors look cohesive and readable in a TUI.
        rgb = colorsys.hls_to_rgb(hue, 0.6, 0.7)

        # Convert RGB (0-1) to Hex strings
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        color_map[label] = hex_color

    return color_map


def segments_to_rich_markup(
        text: str, offsets: List[Tuple[int, int]], labels: List[str], label_colors: Dict[str, str]
) -> str:
    if not offsets:
        return escape(text)

    result = []
    prev = 0
    for (start, end), label in zip(offsets, labels):
        if start > prev:
            result.append(escape(text[prev:start]))

        color = label_colors.get(label, "white")
        seg_text = escape(text[start:end])

        segment_part = f"[black on {color}] {seg_text} [/]"
        tag_part = f" [bold {color}]<[/][{color}]{label}[/][bold {color}]>[/] "
        result.append(f"\n{tag_part}\n{segment_part}")
        prev = end

    if prev < len(text):
        result.append(escape(text[prev:]))

    return "".join(result)


# ────────────────────────────────────────────────
# Widgets
# ────────────────────────────────────────────────

class InputPanel(Vertical):
    def compose(self) -> ComposeResult:
        # yield Label("Reasoning Trace", id="title")

        # Wrap selection in a Vertical container to keep it together
        with Vertical(id="selection-container-aligner"):
            yield Label("Segmentation Method:", id="method-label")
            yield Select(
                [
                    ("Fuzzy", "fuzzy"),
                    ("Graph Maxing", "graph"),
                    ("Union", "merge"),
                    ("Majority Voting", "voting"),
                    ("Flatten", "flatten"),
                    ("Intersection", "intersect"),
                    ("None", "none"),
                ],
                value="voting",
                id="aligner-select"
            )
            OffsetFusionFuzzy,
            OffsetFusionGraph,
            OffsetFusionMerge,
            OffsetFusionVoting,
            OffsetFusionFlatten,
            OffsetFusionIntersect,
            OffsetFusion,
        # Segmentation method selector
        with VerticalScroll(id="selection-container"):
            yield Label("Segmentation Method(s):", id="method-label")
            yield SelectionList(
                ("Rule Based Split", "rule"),
                ("Newline Split", "newline"),
                ("LLM (tok-chunk)", "offset"),
                ("LLM (sent-chunk)", "sent"),
                ("Surprisal", "surprisal"),
                ("Entropy", "entropy"),
                ("Flatness Break", "flatness"),
                ("TopK", "topk"),
                ("Forced Decoder", "forced"),
                ("Zero-Shot", "zero"),
                ("Topic Based", "topic"),
                ("Semantic Shift", "semantic"),
                ("PRM Based", "prm"),
                ("Entailment Based", "entailment"),
                id="method-select",
            )

        # Input field stays below, always visible
        yield Label("Input Reasoning Trace:", id="method-label")
        yield TextArea(id="input", show_line_numbers=False)
        yield Button("Submit → Segment", id="submit", variant="primary")


class ResultsPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static(id="results-table")


# ────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────

class MyApp(App):
    TITLE = "RT-SEG"
    CSS_PATH = f"{bp()}/data/tui_css/app.css"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield InputPanel()
            yield ResultsPanel()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#input", TextArea).focus()

    def on_key(self, event: events.Key) -> None:
        if event.key == "enter" and event.ctrl:
            if self.query_one("#input", TextArea).has_focus:
                self.handle_submit()
                event.prevent_default()

    @on(Button.Pressed, "#submit")
    def handle_submit(self, event=None) -> None:
        textarea = self.query_one("#input", TextArea)
        method_select = self.query_one("#method-select", SelectionList)
        selected_methods = method_select.selected

        aligner_select = self.query_one("#aligner-select", Select)
        selected_aligner = aligner_select.value

        text = textarea.text.rstrip().replace("\r\n", "\n")
        if not text:
            return

        engines = []
        method_dict = {"rule": RTRuleRegex,
                       "newline": RTNewLine,
                       "entropy": RTLLMEntropy,
                       "flatness": RTLLMFlatnessBreak,
                       "forced": RTLLMForcedDecoderBased,
                       "offset": RTLLMOffsetBased,
                       "sent": RTLLMSentBased,
                       "surprisal": RTLLMSurprisal,
                       "topk": RTLLMTopKShift,
                       "zero": RTZeroShotSeqClassification,
                       "topic": RTBERTopicSegmentation,
                       "semantic": RTEmbeddingBasedSemanticShift,
                       "prm": RTPRMBase,
                       "entailment": RTEntailmentBasedSegmentation,}
        for method in selected_methods:
            try:
                engines.append(method_dict[method])
            except KeyError:
                print(f"Method {method} not found.")

        aligner_dict = {"fuzzy": OffsetFusionFuzzy,
                        "graph": OffsetFusionGraph,
                        "merge": OffsetFusionMerge,
                        "voting": OffsetFusionVoting,
                        "flatten": OffsetFusionFlatten,
                        "intersect": OffsetFusionIntersect,
                        "none": None,}
        factory = RTSeg(engines=engines, aligner=aligner_dict[selected_aligner], label_fusion_type="concat")
        offsets, seg_labels = factory(text)

        if set(seg_labels) == {"UNK"}:
            seg_labels = [f"{sl}:{idx}" for idx, sl in enumerate(seg_labels)]

        markup = segments_to_rich_markup(text, offsets, seg_labels, generate_label_colors(seg_labels))
        spaced_markup = "\n\n".join(markup.splitlines())

        results_widget = self.query_one("#results-table", Static)
        results_widget.update(f"[bold underline]Method: {'|'.join(selected_methods)}[/]\n\n{spaced_markup}")

        self.query_one(ResultsPanel).scroll_end()
        textarea.focus()



if __name__ == "__main__":
    MyApp().run()