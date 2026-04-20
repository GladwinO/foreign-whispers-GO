"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Tunable parameters
# =============================================================================

# Spanish TTS speaking rate heuristic, used to convert a time budget (in
# seconds) into a character budget.  ~15 chars/s matches ~4.5 syllables/s
# for Romance languages (Spanish, Italian, French, Portuguese).  English
# is closer to ~17.  Adjust if you target a different language or if
# empirical measurement of your TTS shows a different rate.
_SPANISH_CHARS_PER_SECOND: int = 15

# MarianMT model identifier.  Swap to a different Helsinki-NLP checkpoint
# to target a different language pair.
_MARIAN_MODEL_NAME: str = "Helsinki-NLP/opus-mt-en-es"

# Max input/output token length for MarianMT.  512 comfortably handles any
# single segment in a typical video.
_MARIAN_MAX_TOKEN_LENGTH: int = 512

# Beam width for MarianMT decoding.  Higher = better quality + slower.
# 4 is a common default; 1 is pure greedy (fastest, lowest quality).
_MARIAN_NUM_BEAMS: int = 4

@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str

# =============================================================================
# Ordered so longer phrases are substituted before shorter ones that might
# be substrings.
# =============================================================================
_SPANISH_BREVITY_RULES: list[tuple[str, str]] = [
    # Long connectors → shorter equivalents
    ("a pesar de que", "aunque"),
    ("al mismo tiempo que", "mientras"),
    ("en el caso de que", "si"),
    ("a fin de que", "para que"),
    ("con el fin de", "para"),
    ("de manera que", "así"),
    ("de modo que", "así"),
    ("a menos que", "salvo si"),
    ("sin embargo,", "pero"),
    ("no obstante,", "pero"),

    # Verbose verb phrases → simpler verbs
    ("llevar a cabo", "realizar"),
    ("tomar en cuenta", "considerar"),
    ("dar comienzo a", "comenzar"),
    ("hace referencia a", "se refiere a"),
    ("poner de manifiesto", "mostrar"),
    ("hacer uso de", "usar"),
    ("tener en consideración", "considerar"),
    ("dar lugar a", "causar"),

    # Temporal verbosity
    ("en este momento", "ahora"),
    ("en la actualidad", "hoy"),
    ("por el momento", "por ahora"),
    ("en aquel entonces", "entonces"),

    # Filler removal (inside sentence)
    (", realmente,", ","),
    (", básicamente,", ","),
    (", obviamente,", ","),
    (", en realidad,", ","),

    # Filler removal (mid-sentence, no flanking commas)
    ("realmente ", ""),
    ("básicamente ", ""),
    ("obviamente ", ""),

    # Standard Spanish contractions occasionally missed by MT
    (" de el ", " del "),
    (" a el ", " al "),

    # Gerund forms of the most common verbose verbs
    ("llevando a cabo", "realizando"),
    ("tomando en cuenta", "considerando"),
    ("dando comienzo a", "comenzando"),
    ("haciendo uso de", "usando"),

    # Past tense forms
    ("llevó a cabo", "realizó"),
    ("tomó en cuenta", "consideró"),
]

# =============================================================================
# MarianMT lazy-loaded translator — only initialized when first needed
# =============================================================================
_marian_model = None
_marian_tokenizer = None


def _get_marian_translator():
    """Lazily load the MarianMT en→es model.

    Returns a tuple (model, tokenizer). Loads on first call; cached thereafter.
    """
    global _marian_model, _marian_tokenizer
    if _marian_model is None:
        from transformers import MarianMTModel, MarianTokenizer
        logger.info("Loading MarianMT model %s (first call only)", _MARIAN_MODEL_NAME)
        _marian_tokenizer = MarianTokenizer.from_pretrained(_MARIAN_MODEL_NAME)
        _marian_model = MarianMTModel.from_pretrained(_MARIAN_MODEL_NAME)
    return _marian_model, _marian_tokenizer


def _translate_with_marian(source_text: str) -> str:
    """Translate English → Spanish using MarianMT.

    Returns the translated Spanish text. Raises on failure.
    """
    model, tokenizer = _get_marian_translator()
    inputs = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=_MARIAN_MAX_TOKEN_LENGTH,
    )
    output_ids = model.generate(
        **inputs,
        max_length=_MARIAN_MAX_TOKEN_LENGTH,
        num_beams=_MARIAN_NUM_BEAMS,
        early_stopping=True,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()



# =============================================================================
# Rule-based helper — reports which rules fired for transparent rationales
# =============================================================================
def _apply_brevity_rules(text: str) -> tuple[str, list[str]]:
    """Apply Spanish brevity rules to *text*.

    Returns:
        A tuple ``(result, applied)`` where ``result`` is the transformed text
        and ``applied`` is a list of human-readable rule descriptions for
        every rule that fired.  Returns ``(text, [])`` if no rule matched.
    """
    result = text
    applied: list[str] = []
    for long_phrase, short_phrase in _SPANISH_BREVITY_RULES:
        if long_phrase in result:
            result = result.replace(long_phrase, short_phrase)
            applied.append(
                f"'{long_phrase}' → '{short_phrase}'" if short_phrase
                else f"removed '{long_phrase.strip()}'"
            )

    # Normalize whitespace and punctuation artifacts
    while "  " in result:
        result = result.replace("  ", " ")
    result = result.replace(",,", ",").replace(" ,", ",").replace(" .", ".")
    return result.strip(), applied



def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
     # Parameters reserved for future LLM layer
    _ = context_prev, context_next

    target_chars = int(target_duration_s * _SPANISH_CHARS_PER_SECOND)
    baseline_len = len(baseline_es)
    candidates: list[TranslationCandidate] = []

    # -------------------------------------------------------------------------
    # Layer 1: Rule-based brevity on the baseline translation
    # -------------------------------------------------------------------------
    rule_text, rule_applied = _apply_brevity_rules(baseline_es)
    if rule_applied and len(rule_text) < baseline_len:
        candidates.append(TranslationCandidate(
            text=rule_text,
            char_count=len(rule_text),
            brevity_rationale="Rule-based: " + "; ".join(rule_applied),
        ))

    # -------------------------------------------------------------------------
    # Layer 2: MarianMT alternate translation (+ rules on top)
    # -------------------------------------------------------------------------
    try:
        marian_text = _translate_with_marian(source_text)

        # Plain MarianMT output, if shorter than baseline
        if marian_text and len(marian_text) < baseline_len:
            candidates.append(TranslationCandidate(
                text=marian_text,
                char_count=len(marian_text),
                brevity_rationale="Alternate translation via MarianMT (Helsinki-NLP)",
            ))

        # MarianMT + rule-based
        marian_ruled, marian_rules = _apply_brevity_rules(marian_text)
        if (marian_rules
                and len(marian_ruled) < baseline_len
                and marian_ruled not in (c.text for c in candidates)):
            candidates.append(TranslationCandidate(
                text=marian_ruled,
                char_count=len(marian_ruled),
                brevity_rationale="MarianMT + rules: " + "; ".join(marian_rules),
            ))

    except Exception as exc:
        logger.warning("MarianMT layer failed: %s", exc)

    # -------------------------------------------------------------------------
    # Deduplicate and sort shortest first (per docstring contract)
    # -------------------------------------------------------------------------
    seen: set[str] = set()
    unique: list[TranslationCandidate] = []
    for c in candidates:
        if c.text not in seen:
            seen.add(c.text)
            unique.append(c)
    unique.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: %d candidate(s) for %.1fs budget "
        "(target=%d chars, baseline=%d chars)",
        len(unique), target_duration_s, target_chars, baseline_len,
    )
    return unique
