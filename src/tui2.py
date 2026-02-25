import random
import traceback
from typing import List, Tuple, Dict, Literal
import colorsys
from rich.text import Text
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on, events
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, Select, SelectionList, RadioButton, RadioSet

from rt_segmentation import (RTLLMOffsetBased,
                             RTLLMForcedDecoderBased,
                             RTLLMSegUnitBased,
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
                             RTEmbeddingBasedSemanticShift,
                             RTPRMBase,
                             RTEntailmentBasedSegmentation,
                             RTLLMReasoningFlow,
                             RTLLMThoughtAnchor,
                             RTLLMArgument,
                             RTZeroShotSeqClassificationTA,
                             RTZeroShotSeqClassificationRF,
                             RTSeg,
                             OffsetFusionFuzzy,
                             OffsetFusionGraph,
                             OffsetFusionMerge,
                             OffsetFusionVoting,
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
        seg_text = escape(text[start:end]).strip()

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

class InputPanel1(Vertical):
    def compose(self) -> ComposeResult:
        with Vertical(classes="binary-col"):
            yield Label("Label Fusion Mode:")
            with RadioSet(id="label-fusion"):
                yield RadioButton("Concat", id="concat")
                yield RadioButton("Majority", id="majority")

        with Vertical(classes="binary-col"):
            yield Label("Seg Base Unit:")
            with RadioSet(id="seg-unit"):
                yield RadioButton("Clause", id="clause")
                yield RadioButton("Sentence", id="sent")

        with Vertical(id="selection-container-aligner"):
            yield Label("Offset Late Fusion(s):", id="method-label")
            yield Select(
                [
                    ("Fuzzy", "fuzzy"),
                    ("Graph Maxing", "graph"),
                    ("Union", "merge"),
                    ("Majority Voting", "voting"),
                    ("Intersection", "intersect"),
                    ("None", "none"),
                ],
                value="voting",
                id="aligner-select",
            )

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
                ("LLM (reasoning-flow)", "llmrf"),
                ("LLM (thought-anchor)", "llmta"),
                ("LLM (argument)", "llmarg"),
                ("Zero-Shot (reasoning-flow)", "zerorf"),
                ("Zero-Shot (thought-anchor)", "zerota"),
                id="method-select",
            )


class InputPanel2(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Input Reasoning Trace:", id="method-label")
        yield TextArea(id="input", show_line_numbers=False)


class ResultsPanel(VerticalScroll):
    def compose(self) -> ComposeResult:
        yield Static(id="results-table")


class MiddlePanel(Vertical):
    """Results in the middle, submit button centered beneath."""
    def compose(self) -> ComposeResult:
        yield ResultsPanel()
        yield Button("Submit → Segment", id="submit", variant="primary")


# ────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────

class MyApp(App):
    TITLE = "RT-SEG"
    CSS_PATH = f"{bp()}/data/tui_css/app2.css"

    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            yield InputPanel1()
            yield MiddlePanel()
            yield InputPanel2()

    def on_mount(self) -> None:
        # Text Input Area:
        self.query_one("#input", TextArea).focus()
        self.query_one("#label-fusion", RadioSet).value = "concat"
        self.query_one("#seg-unit", RadioSet).value = "clause"

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
                       "unit": RTLLMSegUnitBased,
                       "surprisal": RTLLMSurprisal,
                       "topk": RTLLMTopKShift,
                       "zero": RTZeroShotSeqClassification,
                       "topic": RTBERTopicSegmentation,
                       "semantic": RTEmbeddingBasedSemanticShift,
                       "prm": RTPRMBase,
                       "entailment": RTEntailmentBasedSegmentation,
                       "llmrf": RTLLMReasoningFlow,
                       "llmta": RTLLMThoughtAnchor,
                       "llmarg": RTLLMArgument,
                       "zerota": RTZeroShotSeqClassificationTA,
                       "zerorf": RTZeroShotSeqClassificationRF,
        }
        for method in selected_methods:
            try:
                engines.append(method_dict[method])
            except KeyError:
                print(f"Method {method} not found.")

        fusion_radio = self.query_one("#label-fusion", RadioSet)
        unit_radio = self.query_one("#seg-unit", RadioSet)

        selected_fusion = (
            fusion_radio.pressed_button.id
            if fusion_radio.pressed_button
            else "concat"
        )

        selected_unit = (
            unit_radio.pressed_button.id
            if unit_radio.pressed_button
            else "clause"
        )

        aligner_dict = {"fuzzy": OffsetFusionFuzzy,
                        "graph": OffsetFusionGraph,
                        "merge": OffsetFusionMerge,
                        "voting": OffsetFusionVoting,
                        "intersect": OffsetFusionIntersect,
                        "none": None,}

        factory = RTSeg(engines=engines,
                        aligner=aligner_dict[selected_aligner],
                        label_fusion_type=selected_fusion,
                        seg_base_unit=selected_unit)
        try:
            offsets, seg_labels = factory(text)
            if set(seg_labels) == {"UNK"}:
                seg_labels = [f"{sl}:{idx}" for idx, sl in enumerate(seg_labels)]

            markup = segments_to_rich_markup(text, offsets, seg_labels, generate_label_colors(seg_labels))
            spaced_markup = "\n\n".join(markup.splitlines())

            results_widget = self.query_one("#results-table", Static)
            results_widget.update(f"[bold underline]Method: {'|'.join(selected_methods)}[/]\n\n{spaced_markup}")

            self.query_one(ResultsPanel).scroll_end()
            textarea.focus()
        except Exception as e:
            error_str = traceback.format_exc()
            results_widget = self.query_one("#results-table", Static)
            results_widget.update(
                "[bold red]Exception occurred:[/]\n\n"
                + escape(error_str)  # IMPORTANT: prevents Rich markup parsing
            )

            self.query_one(ResultsPanel).scroll_end()
            textarea.focus()




if __name__ == "__main__":
    MyApp().run()