import random
from typing import List, Tuple, Dict, Literal
import colorsys
from rich.text import Text
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual import on, events
from textual.widgets import Header, Footer, Static, Button, TextArea, Label, Select

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
                             load_example_trace, RTLLMSurprisal, RTLLMEntropy, RTLLMTopKShift, RTLLMFlatnessBreak,
                             export_gold_set)


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
        result.append(f"{tag_part}{segment_part}")
        prev = end

    if prev < len(text):
        result.append(escape(text[prev:]))

    return "".join(result)


# ────────────────────────────────────────────────
# Widgets
# ────────────────────────────────────────────────

class InputPanel(Vertical):
    def compose(self) -> ComposeResult:
        yield Label("Reasoning Trace", id="title")

        # Wrap selection in a Vertical container to keep it together
        with Vertical(id="selection-container"):
            yield Label("Segmentation Method:", id="method-label")
            yield Select(
                [
                    ("Rule Based Split", "rule"),
                    ("Newline Split", "newline"),
                    ("LLM (tok-chunk)", "offset"),
                    ("LLM (sent-chunk)", "sent"),
                    ("LLM Surprisal", "surprisal"),
                    ("LLM Entropy", "entropy"),
                    ("LLM Flatness Break", "flatness"),
                    ("LLM TopK", "topk"),
                    ("LLM Forced Decoder", "forced"),
                    ("LLM Zero-Shot", "zero"),
                    ("Topic Based", "topic"),
                ],
                value="rule",
                id="method-select"
            )

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
        method_select = self.query_one("#method-select", Select)
        selected_method = method_select.value

        text = textarea.text.rstrip().replace("\r\n", "\n")
        if not text:
            return
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        # Routing Logic
        if selected_method == "rule":
            offsets, seg_labels = RTRuleRegex._segment(trace=text)
        elif selected_method == "newline":
            offsets, seg_labels = RTNewLine._segment(trace=text)
        elif selected_method == "entropy":
            offsets, seg_labels = RTLLMEntropy._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_surprisal"),
                model_name=model_name)
        elif selected_method == "flatness":
            offsets, seg_labels = RTLLMFlatnessBreak._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_surprisal"),
                model_name=model_name)
        elif selected_method == "forced":
            offsets, seg_labels = RTLLMForcedDecoderBased._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_forceddecoder"),
                model_name="Qwen/Qwen2.5-7B-Instruct")
        elif selected_method == "offset":
            offsets, seg_labels = RTLLMOffsetBased._segment(
                trace=text,
                chunk_size=20,
                prompt="",
                system_prompt=load_prompt("system_prompt_offset"),
                model_name=model_name
                )
        elif selected_method == "sent":
            offsets, seg_labels = RTLLMSentBased._segment(
                trace=text,
                chunk_size=20,
                prompt="",
                system_prompt=load_prompt("system_prompt_sentbased"),
                model_name=model_name
                )
        elif selected_method == "surprisal":
            offsets, seg_labels = RTLLMSurprisal._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_surprisal"),
                model_name=model_name)
        elif selected_method == "topk":
            offsets, seg_labels = RTLLMTopKShift._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_surprisal"),
                model_name=model_name)
        elif selected_method == "zero":
            offsets, seg_labels = RTZeroShotSeqClassification._segment(
                trace=text,
                model_name="facebook/bart-large-mnli")
        elif selected_method == "topic":
            offsets, seg_labels = RTBERTopicSegmentation._segment(
                trace=text,
                system_prompt=load_prompt("system_prompt_topic_label"),
                model_name=model_name,
                all_custom_labels=True)
        else:
            raise NotImplementedError(f"Method {selected_method} not implemented.")

        if set(seg_labels) == {"UNK"}:
            seg_labels = [f"{sl}:{idx}" for idx, sl in enumerate(seg_labels)]

        markup = segments_to_rich_markup(text, offsets, seg_labels, generate_label_colors(seg_labels))
        spaced_markup = "\n\n".join(markup.splitlines())

        results_widget = self.query_one("#results-table", Static)
        results_widget.update(f"[bold underline]Method: {selected_method}[/]\n\n{spaced_markup}")

        self.query_one(ResultsPanel).scroll_end()
        textarea.focus()



if __name__ == "__main__":
    MyApp().run()