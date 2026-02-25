import subprocess
import sys
import re
import spacy
from functools import lru_cache

import stanza



class UnitSegmentor:
    patterns = {
            "display_math": r"\$\$.*?\$\$",
            "inline_math": r"\$.*?\$",
            "code_block": r"```.*?```",
            "inline_code": r"`.*?`"
        }

    @staticmethod
    @lru_cache(maxsize=1)
    def load_stanza_constituency():
        """
        Load or initialize the Stanza pipeline with constituency parsing.
        """
        stanza.download('en')
        return stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,constituency',
            verbose=False
        )

    @staticmethod
    def get_math_aware_clauses(text: str) -> list[tuple[int, int]]:
        placeholders = []

        def mask_block(match):
            ph = f"MATHBLOCK{len(placeholders)}Z"
            placeholders.append(match.group())
            return ph

        combined = re.compile(f"({'|'.join(UnitSegmentor.patterns.values())})", re.DOTALL)
        masked_text = combined.sub(mask_block, text)

        nlp = UnitSegmentor.load_stanza_constituency()
        doc = nlp(masked_text)

        clausal_labels = {"S", "SBAR", "SBARQ", "SINV", "SQ"}
        final_offsets = []

        for sent in doc.sentences:
            words = sent.words

            def assign_indices(node, current_idx):
                if node.is_leaf():
                    node.word_idx = current_idx
                    node.all_indices = {current_idx}
                    return current_idx + 1
                indices = set()
                for child in node.children:
                    current_idx = assign_indices(child, current_idx)
                    indices.update(child.all_indices)
                node.all_indices = indices
                return current_idx

            assign_indices(sent.constituency, 0)

            split_word_indices = {0}

            def find_major_splits(node):
                found_clauses = 0
                for child in node.children:
                    if child.label in clausal_labels:
                        # HEURISTIC: Only split if the clause is large enough (e.g., > 3 words)
                        # This prevents splitting "So," or "since b" into their own segments.
                        if len(child.all_indices) > 3:
                            # We only split if this is a secondary clause in this branch
                            found_clauses += 1
                            if found_clauses > 1 or child.label == "SBAR":
                                split_word_indices.add(min(child.all_indices))

                    if child.label in {":", ";"}:
                        split_word_indices.add(min(child.all_indices))

                    if not child.is_leaf():
                        find_major_splits(child)

            find_major_splits(sent.constituency)

            sorted_splits = sorted(list(split_word_indices))
            for i in range(len(sorted_splits)):
                start_w_idx = sorted_splits[i]
                end_w_idx = sorted_splits[i + 1] - 1 if i + 1 < len(sorted_splits) else len(words) - 1

                if start_w_idx > end_w_idx: continue

                start_off = words[start_w_idx].start_char
                end_off = words[end_w_idx].end_char

                if re.search(r'\w', masked_text[start_off:end_off]):
                    final_offsets.append((start_off, end_off))

        # Sort and merge overlaps
        final_offsets.sort()
        unique_offsets = []
        if final_offsets:
            unique_offsets.append(final_offsets[0])
            for current in final_offsets[1:]:
                prev_start, prev_end = unique_offsets[-1]
                curr_start, curr_end = current
                if curr_start >= prev_end:
                    unique_offsets.append(current)
                else:
                    unique_offsets[-1] = (prev_start, max(prev_end, curr_end))

        if not unique_offsets: return []

        # CONTIGUOUS MAPPING: Ensuring no gaps
        contig = []
        for idx in range(len(unique_offsets) - 1):
            contig.append((unique_offsets[idx][0], unique_offsets[idx + 1][0]))
        contig.append((unique_offsets[-1][0], len(text)))

        # FINAL PASS: Merge "Micro-segments" (segments shorter than 8 characters or 2 words)
        # This fixes the "So," and "since b" orphan problem.
        refined = []
        for start, end in contig:
            seg_text = text[start:end].strip()
            word_count = len(seg_text.split())

            if refined and (word_count < 3 or len(seg_text) < 10):
                # Merge with previous
                prev_start, _ = refined.pop()
                refined.append((prev_start, end))
            else:
                refined.append((start, end))

        return refined


    @staticmethod
    @lru_cache(maxsize=1)
    def load_spacy_model(model_name="en_core_web_sm"):
        """
        Checks if a spaCy model is installed. If not, downloads it.
        Returns the loaded nlp object.
        """
        if not spacy.util.is_package(model_name):
            print(f"Model '{model_name}' not found. Downloading...")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                print(f"Successfully installed '{model_name}'.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download model: {e}")
                return None

        return spacy.load(model_name)

    @staticmethod
    def get_math_aware_clauses_dep(text: str, min_words=4):
        combined_pattern = f"({'|'.join(UnitSegmentor.patterns.values())})"
        matches = list(re.finditer(combined_pattern, text, flags=re.DOTALL))
        masked_text = text
        placeholders = {}

        for i, match in enumerate(reversed(matches)):
            placeholder = f"BLOCK_ID_{len(matches) - 1 - i}_"
            start, end = match.span()
            placeholders[placeholder] = match.group()
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

        nlp = UnitSegmentor.load_spacy_model()
        if not nlp:
            return []

        if "sentencizer" not in nlp.pipe_names:
            sentencizer = nlp.create_pipe("sentencizer")
            nlp.add_pipe(sentencizer, before="parser")

        doc = nlp(masked_text)

        clause_labels = {'ccomp', 'advcl', 'relcl', 'acl', 'csubj'}
        all_clauses = []

        for sent in doc.sents:
            sent_start, sent_end = sent.start_char, sent.end_char
            split_indices = []

            for token in sent:
                token_phrase = ' '.join([t.text.lower() for t in token.subtree])
                if re.search(r"\blet('s| us| me)?\b", token_phrase):
                    continue

                if ((token.dep_ in clause_labels and token.pos_ in {'VERB', 'AUX'}) or
                        token.dep_ in {'cc', 'mark'}):
                    start_idx = token.idx
                    prev_text = masked_text[sent_start:start_idx].strip()
                    if len(prev_text.split()) >= min_words:
                        split_indices.append(start_idx)

                if token.text == ',' and token.i + 1 < len(sent):
                    next_tok = sent[token.i - sent.start + 1]
                    if next_tok.pos_ in {'VERB', 'AUX'} or next_tok.dep_ == 'mark':
                        start_idx = token.idx + 1
                        prev_text = masked_text[sent_start:start_idx].strip()
                        if len(prev_text.split()) >= min_words:
                            split_indices.append(start_idx)

            boundaries = sorted(list(set([sent_start] + split_indices + [sent_end])))

            for i in range(len(boundaries) - 1):
                s, e = boundaries[i], boundaries[i + 1]
                segment = masked_text[s:e].strip()
                if not segment:
                    continue

                for ph, original_val in placeholders.items():
                    segment = segment.replace(ph, original_val)

                if re.search(r'\w+', segment):
                    all_clauses.append((s, e, segment))

        return all_clauses

    @staticmethod
    def get_math_aware_sents(text: str):
        # 1. Identify all protected spans (Math/Code)
        combined_pattern = f"({'|'.join(UnitSegmentor.patterns.values())})"
        matches = list(re.finditer(combined_pattern, text, flags=re.DOTALL))

        # 2. Create a masked version of the text
        masked_text = text
        placeholders = {}

        # We replace from back to front to keep indices valid during string manipulation
        for i, match in enumerate(reversed(matches)):
            placeholder = f"BLOCK_ID_{len(matches) - 1 - i}_"
            start, end = match.span()
            placeholders[placeholder] = match.group()
            masked_text = masked_text[:start] + placeholder + masked_text[end:]

        # 3. Process with spaCy
        doc = UnitSegmentor.load_spacy_model()(masked_text)

        segments = []
        for sent in doc.sents:
            sent_text = sent.text

            # 4. Restore the original content (Unmasking)
            for placeholder, original in placeholders.items():
                if placeholder in sent_text:
                    sent_text = sent_text.replace(placeholder, original)

            segments.append((sent.start_char, sent.end_char))

        return segments

if __name__ == "__main__":
    # Test
    t = """Okay, let's try to figure out this problem. The question is asking for the sum of all integer bases b > 9 where 17_b is a divisor of 97_b. Hmm, first I need to understand what these numbers mean in different bases.\n\nSo, in base b, the number 17_b would be equal to 1*b^1 + 7*b^0, which simplifies to b + 7. Similarly, 97_b is 9*b^1 + 7*b^0, so that's 9b + 7. The problem states that 17_b divides 97_b, meaning that (9b + 7) divided by (b + 7) should be an integer. So, I need to find all bases b > 9 where (9b + 7)/(b + 7) is an integer.\n\nLet me write that division as a fraction: (9b + 7)/(b + 7). Maybe I can simplify this expression. Let's try polynomial division or see if I can rewrite it. Let me see:\n\nIf I divide 9b + 7 by b + 7, maybe I can do some algebra. Let's see:\n\nLet me consider 9b + 7 divided by b + 7. Let's write it as 9*(b + 7) - 63 + 7 divided by (b + 7). Wait, that might not be helpful. Alternatively, maybe express the numerator in terms of the denominator.\n\nLet me think. Let me write 9b + 7 as 9*(b + 7) - 63 + 7. Wait, 9*(b + 7) is 9b + 63. So 9b +7 = 9*(b +7) - 56. Therefore, (9b +7)/(b +7) = 9 - 56/(b +7). So the expression simplifies to 9 - 56/(b +7). For this to be an integer, 56/(b +7) must be an integer. Therefore, (b +7) must be a divisor of 56.\n\nSo, since b > 9, then b +7 must be a positive divisor of 56, and also since b +7 must be greater than 9 +7 = 16? Wait, no. Wait, b is greater than 9, so b +7 is greater than 16. Therefore, we need to find all positive divisors of 56 that are greater than 16, and then subtract 7 from them to get b.\n\nWait, but let me check. Let me list all divisors of 56. The divisors of 56 are 1, 2, 4, 7, 8, 14, 28, 56. But since b +7 must be greater than 16 (since b >9), then the possible divisors are 28, 56. Because 14 is 14, which would make b +7=14 → b=7, but b must be greater than 9. So 14 is too small. Similarly, 8 and 7 are too small. So the only divisors of 56 greater than 16 are 28 and 56. Therefore, b +7 =28 or 56. Therefore, b=21 or b=56-7=49. Wait, 56-7 is 49. So the possible bases are 21 and 49. Therefore, the sum is 21 +49=70.\n\nWait, but let me check if that's correct. Let me verify each case.\n\nFirst, for b=21: 17_b is 1*21 +7=28. 97_b is 9*21 +7=198. 198 divided by 28 is 7. So 198/28=7. So yes, that works.\n\nFor b=49: 17_b is 1*49 +7=56. 97_b is 9*49 +7=441 +7=448. 448 divided by 56 is 8. So that's also an integer. So both cases work.\n\nAre there any other divisors? Let me check again. The divisors of 56 are 1,2,4,7,8,14,28,56. So the ones greater than 16 are 28 and 56. So yes, only those two. Therefore, the answer is 21 +49=70. So the sum is 70.\n\nBut wait, let me make sure there are no other divisors. For example, if b +7 is a divisor of 56, but maybe negative divisors? But since b>9, b+7>16, so divisors have to be positive. So no negative divisors. So yes, only 28 and 56. So the answer is 21 +49=70.\n\nTherefore, the final answer is 70. Let me check if that's correct. Yes, seems right."""
    # t = "This is a test sentence, while i try not to spoil the whole story. Please, forgive me on this. I like houses, which are not too big."
    # segments = UnitSegmentor.get_math_aware_sents(t)
    segments = UnitSegmentor.get_math_aware_clauses(t)
    for idx, c in enumerate(segments):
        print(f"{idx} :: {c[0], c[1]} :: ", t[c[0]:c[1]])
    print(segments)
    """for idx, c in enumerate(segments):
        print(f"{idx} :: {c[0], c[1]} :: ", t[c[0]:c[1]])"""