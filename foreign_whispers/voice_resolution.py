"""Voice resolution for Chatterbox speaker cloning.

Resolves which reference WAV to use for a given target language
and optional speaker ID. The Chatterbox container expects a filename
relative to its /app/voices/ mount point.
"""

from pathlib import Path


def resolve_speaker_wav(
    speakers_dir: Path,
    target_language: str,
    speaker_id: str | None = None,
) -> str:
    """Resolve the reference WAV path for voice cloning.

    Resolution order:
    1. Speaker-specific: ``speakers/{lang}/{speaker_id}.wav``
    2. Language default: ``speakers/{lang}/default.wav``
    3. Global default:   ``speakers/default.wav``

    Args:
        speakers_dir: Absolute path to the ``pipeline_data/speakers/`` directory.
        target_language: ISO 639-1 language code (e.g. ``"es"``, ``"fr"``).
        speaker_id: Optional speaker label (e.g. ``"SPEAKER_00"``).
            If None, skips straight to language default.

    Returns:
        Relative path string suitable for passing to ChatterboxClient
        (relative to ``speakers_dir``).
    """
    lang_dir = speakers_dir / target_language

    # 1. Speaker-specific WAV
    if speaker_id:
        speaker_wav = lang_dir / f"{speaker_id}.wav"
        if speaker_wav.exists():
            return f"{target_language}/{speaker_id}.wav"

    # 2. Language default
    lang_default = lang_dir / "default.wav"
    if lang_default.exists():
        return f"{target_language}/default.wav"

    # 3. Global default
    return "default.wav"