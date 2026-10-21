import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from transcriber import TranscriptionResult, TranscriptionSegment, clean_transcript, format_timestamped_transcript


PROMPT_LEAK_PHRASES = (
    "\u0644\u0627 \u062a\u062a\u0631\u062c\u0645",
    "\u0627\u0643\u062a\u0628 \u0627\u0644\u0643\u0644\u0627\u0645 \u0628\u0646\u0641\u0633 \u0627\u0644\u0644\u063a\u0629",
    "\u0627\u0643\u062a\u0628 \u0627\u0644\u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629 \u0628\u062d\u0631\u0648\u0641 \u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629",
    "Latin letters",
    "Do not translate",
    "Transcribe exactly what is spoken",
    "\u0638\u201e\u0637\u00a7 \u0637\u06be\u0637\u06be\u0637\u00b1\u0637\u00ac\u0638\u2026",
    "\u0637\u00a7\u0638\u0192\u0637\u06be\u0637\u00a8 \u0637\u00a7\u0638\u201e\u0638\u0192\u0638\u201e\u0637\u00a7\u0638\u2026",
    "\u0637\u00a8\u0637\u00ad\u0637\u00b1\u0638\u2020\u0638\u06be \u0637\u00a5\u0638\u2020\u0637\u00ac\u0638\u201e\u0638\u0639\u0637\u00b2\u0638\u0639\u0637\u00a9",
)
GENERIC_HALLUCINATION_PHRASES = (
    "welcome to my channel",
    "please turn on subtitles",
    "subtitles from the settings",
    "in this video",
    "today we prepare",
    "i will show you",
    "let's prepare",
    "lets prepare",
    "the ingredients",
    "thank you for watching",
    "don't forget to subscribe",
    "dont forget to subscribe",
    "like and subscribe",
    "click the bell",
    "this video is sponsored",
    "we will explain how to make",
    "\u0637\u062a\u0631\u062c\u0645\u0629 \u0646\u0627\u0646\u0633\u064a \u0642\u0646\u0642\u0631",
    "\u0627\u0634\u062a\u0631\u0643\u0648\u0627 \u0641\u064a \u0627\u0644\u0642\u0646\u0627\u0629",
    "\u0637\u00a7\u0637\u00b4\u0637\u06be\u0637\u00b1\u0643\u0638\u02c6\u0637\u00a7 \u0638\u067e\u0638\u0160 \u0637\u00a7\u0638\u201e\u0638\u201a\u0638\u2020\u0637\u00a7\u0637\u00a9",
)


@dataclass(frozen=True)
class TranscriptQcSettings:
    enabled: bool = True
    min_meaningful_words: int = 3
    prompt_leak_rejection: bool = True
    repetition_rejection: bool = True
    initial_prompt: str | None = None
    mixed_language_protection: bool = True
    language_rejection_soft: bool = True
    allowed_languages: frozenset[str] = frozenset({"ar", "en"})
    vad_speech_ratio: float | None = None


@dataclass(frozen=True)
class TranscriptQcResult:
    passed: bool
    reason: str | None = None
    result: TranscriptionResult | None = None
    confidence: str = "medium"
    suspicious_signals: list[str] | None = None
    protective_signals: list[str] | None = None
    qc_severity: str = "pass"


def validate_transcription(
    result: TranscriptionResult,
    *,
    duration_seconds: float | None,
    settings: TranscriptQcSettings,
) -> TranscriptQcResult:
    if not settings.enabled:
        return TranscriptQcResult(
            passed=True,
            reason=None,
            result=result,
            confidence="medium",
            suspicious_signals=[],
            protective_signals=["qc_skipped"],
            qc_severity="pass",
        )

    text = clean_transcript(result.text or "")
    suspicious_signals = collect_suspicious_signals(text, result, duration_seconds, settings)
    protective_signals = collect_protective_signals(text, result, duration_seconds, settings)

    if not text:
        return qc_fail("empty_transcript", suspicious_signals, protective_signals)

    if settings.prompt_leak_rejection and contains_prompt_leak(text, settings.initial_prompt):
        return qc_fail("prompt_leak_detected", suspicious_signals, protective_signals)

    if segment_duration_exceeds_audio(result.segments, duration_seconds):
        return qc_fail("segment_duration_exceeds_audio", suspicious_signals, protective_signals)

    if transcript_audio_length_mismatch(text, duration_seconds):
        return qc_fail("transcript_audio_length_mismatch", suspicious_signals, protective_signals)

    if contains_generic_hallucination_phrase(text, duration_seconds, result, settings):
        return qc_fail("generic_hallucination_phrase", suspicious_signals, protective_signals)

    if settings.repetition_rejection and is_repetitive_hallucination(text):
        if is_soft_repetition_warning(text, duration_seconds, suspicious_signals, protective_signals):
            return qc_pass(
                result,
                reason="repeated_hallucination",
                confidence="medium",
                suspicious_signals=suspicious_signals,
                protective_signals=protective_signals,
                qc_severity="soft_warning",
            )
        return qc_fail("repeated_hallucination", suspicious_signals, protective_signals)

    if is_short_generic_hallucination(text, duration_seconds, result.language_probability):
        return qc_fail("short_generic_hallucination", suspicious_signals, protective_signals)

    filtered_segments, weak_segment_ratio = filter_reliable_segments(result.segments)
    if result.segments and not filtered_segments:
        return qc_fail("low_confidence_segments", suspicious_signals, protective_signals)

    filtered_text = text
    filtered_result = result
    if filtered_segments and len(filtered_segments) < len(result.segments):
        filtered_text = clean_transcript(" ".join(segment.text for segment in filtered_segments))
        filtered_result = TranscriptionResult(
            text=filtered_text,
            formatted_text=format_timestamped_transcript(filtered_segments),
            segments=filtered_segments,
            detected_language=result.detected_language,
            language_probability=result.language_probability,
        )

    if weak_segment_ratio >= 0.75 and meaningful_word_count(filtered_text) < settings.min_meaningful_words:
        return qc_fail("low_confidence_segments", suspicious_signals, protective_signals)

    if is_too_short(filtered_text, duration_seconds, settings.min_meaningful_words):
        return qc_fail("too_short_transcript", suspicious_signals, protective_signals)

    reason = None
    detected_language = (result.detected_language or "").lower()
    if detected_language and detected_language not in settings.allowed_languages:
        reason = "language_suspicious_but_text_valid"
    severity = "soft_warning" if reason else "pass"
    return TranscriptQcResult(
        passed=True,
        reason=reason,
        result=filtered_result,
        confidence=confidence_for(suspicious_signals, protective_signals),
        suspicious_signals=suspicious_signals,
        protective_signals=protective_signals,
        qc_severity=severity,
    )


def contains_prompt_leak(text: str, initial_prompt: str | None) -> bool:
    normalized_text = normalize_for_match(text)
    for phrase in PROMPT_LEAK_PHRASES:
        if normalize_for_match(phrase) in normalized_text:
            return True

    prompt = (initial_prompt or "").strip()
    if not prompt:
        return False

    normalized_prompt = normalize_for_match(prompt)
    if normalized_prompt and normalized_prompt in normalized_text:
        return True

    if len(normalized_text) >= 24 and len(normalized_prompt) >= 24:
        similarity = SequenceMatcher(None, normalized_text, normalized_prompt).ratio()
        if similarity >= 0.72:
            return True

    prompt_words = meaningful_words(prompt)
    text_words = meaningful_words(text)
    if len(prompt_words) >= 6 and text_words:
        overlap = sum(1 for word in text_words if word in prompt_words)
        if overlap / max(1, len(text_words)) >= 0.7:
            return True

    return False


def segment_duration_exceeds_audio(segments: list[TranscriptionSegment], duration_seconds: float | None) -> bool:
    if duration_seconds is None or not segments:
        return False
    return max(segment.end for segment in segments) > duration_seconds + 2.0


def transcript_audio_length_mismatch(text: str, duration_seconds: float | None) -> bool:
    if duration_seconds is None or duration_seconds <= 0:
        return False
    word_count = meaningful_word_count(text)
    if word_count / duration_seconds > 4.5:
        return True
    return duration_seconds < 15 and word_count > 45


def contains_generic_hallucination_phrase(
    text: str,
    duration_seconds: float | None,
    result: TranscriptionResult,
    settings: TranscriptQcSettings,
) -> bool:
    normalized = normalize_for_match(text)
    matched = any(normalize_for_match(phrase) in normalized for phrase in GENERIC_HALLUCINATION_PHRASES)
    if not matched:
        return False

    weak_segments = False
    if result.segments:
        _, weak_ratio = filter_reliable_segments(result.segments)
        weak_segments = weak_ratio >= 0.5
    suspicious_language = (result.detected_language or "").lower() not in {"ar", "en"}
    weak_language = result.language_probability is None or result.language_probability < 0.70
    weak_vad = settings.vad_speech_ratio is not None and settings.vad_speech_ratio < 0.04
    short_audio = duration_seconds is not None and duration_seconds < 30
    short_text = meaningful_word_count(text) <= 14
    no_real_context = not (contains_arabic_text(text) or contains_latin_words(text))
    return (
        (short_audio and short_text)
        or (suspicious_language and (weak_vad or weak_segments or short_text))
        or (weak_language and (weak_vad or weak_segments))
        or no_real_context
    )


def collect_suspicious_signals(
    text: str,
    result: TranscriptionResult,
    duration_seconds: float | None,
    settings: TranscriptQcSettings,
) -> list[str]:
    signals: list[str] = []
    detected_language = (result.detected_language or "").lower()
    if not text:
        signals.append("empty_transcript")
    if detected_language and detected_language not in settings.allowed_languages:
        signals.append("unsupported_or_unreliable_detected_language")
    if contains_prompt_leak(text, settings.initial_prompt):
        signals.append("prompt_leak_detected")
    if segment_duration_exceeds_audio(result.segments, duration_seconds):
        signals.append("segment_duration_exceeds_audio")
    if transcript_audio_length_mismatch(text, duration_seconds):
        signals.append("transcript_audio_length_mismatch")
    if settings.repetition_rejection and is_repetitive_hallucination(text):
        signals.append("repeated_hallucination")
    if contains_generic_hallucination_phrase(text, duration_seconds, result, settings):
        signals.append("generic_hallucination_phrase")
    if is_short_generic_hallucination(text, duration_seconds, result.language_probability):
        signals.append("short_generic_hallucination")
    if result.segments:
        _, weak_ratio = filter_reliable_segments(result.segments)
        if weak_ratio >= 0.75:
            signals.append("low_confidence_segments")
    if settings.vad_speech_ratio is not None and settings.vad_speech_ratio < 0.02:
        signals.append("weak_vad_speech_ratio")
    if not contains_arabic_text(text) and not contains_latin_words(text):
        signals.append("no_arabic_or_latin_meaningful_text")
    return signals


def collect_protective_signals(
    text: str,
    result: TranscriptionResult,
    duration_seconds: float | None,
    settings: TranscriptQcSettings,
) -> list[str]:
    signals: list[str] = []
    word_count = meaningful_word_count(text)
    if contains_arabic_text(text):
        signals.append("contains_arabic_text")
    if contains_latin_words(text):
        signals.append("contains_latin_words")
    if contains_arabic_text(text) and contains_latin_words(text):
        signals.append("mixed_language_text_detected")
    if not segment_duration_exceeds_audio(result.segments, duration_seconds):
        signals.append("sane_timestamps")
    if word_count >= settings.min_meaningful_words or (duration_seconds is not None and duration_seconds <= 5 and word_count > 0):
        signals.append("meaningful_word_count_ok")
    if not contains_prompt_leak(text, settings.initial_prompt):
        signals.append("no_prompt_leak")
    if not is_repetitive_hallucination(text):
        signals.append("no_repetition")
    if settings.vad_speech_ratio is not None and settings.vad_speech_ratio >= 0.02:
        signals.append("vad_speech_present")
    return signals


def qc_fail(
    reason: str,
    suspicious_signals: list[str],
    protective_signals: list[str],
) -> TranscriptQcResult:
    return TranscriptQcResult(
        passed=False,
        reason=reason,
        result=None,
        confidence="low",
        suspicious_signals=suspicious_signals,
        protective_signals=protective_signals,
        qc_severity="hard_reject",
    )


def qc_pass(
    result: TranscriptionResult,
    *,
    reason: str | None,
    confidence: str,
    suspicious_signals: list[str],
    protective_signals: list[str],
    qc_severity: str,
) -> TranscriptQcResult:
    return TranscriptQcResult(
        passed=True,
        reason=reason,
        result=result,
        confidence=confidence,
        suspicious_signals=suspicious_signals,
        protective_signals=protective_signals,
        qc_severity=qc_severity,
    )


def confidence_for(suspicious_signals: list[str], protective_signals: list[str]) -> str:
    strong_rejections = {
        "prompt_leak_detected",
        "segment_duration_exceeds_audio",
        "transcript_audio_length_mismatch",
        "generic_hallucination_phrase",
        "repeated_hallucination",
        "low_confidence_segments",
    }
    if any(signal in strong_rejections for signal in suspicious_signals):
        return "medium"
    if len(protective_signals) >= 4 and len(suspicious_signals) <= 1:
        return "high"
    if len(suspicious_signals) >= 3:
        return "low"
    return "medium"


def filter_reliable_segments(segments: list[TranscriptionSegment]) -> tuple[list[TranscriptionSegment], float]:
    if not segments:
        return [], 0.0

    reliable: list[TranscriptionSegment] = []
    weak_count = 0
    for segment in segments:
        text = clean_transcript(segment.text or "")
        weak_signals = 0
        available_signals = 0

        if segment.no_speech_prob is not None:
            available_signals += 1
            if segment.no_speech_prob > 0.75:
                weak_signals += 1
        if segment.avg_logprob is not None:
            available_signals += 1
            if segment.avg_logprob < -1.2:
                weak_signals += 1
        if segment.compression_ratio is not None:
            available_signals += 1
            if segment.compression_ratio > 2.4:
                weak_signals += 1
        if meaningful_word_count(text) == 0:
            available_signals += 1
            weak_signals += 1

        is_weak = available_signals > 0 and weak_signals >= max(1, available_signals - 1)
        if is_weak:
            weak_count += 1
        else:
            reliable.append(segment)

    return reliable, weak_count / len(segments)


def is_too_short(text: str, duration_seconds: float | None, min_meaningful_words: int) -> bool:
    if duration_seconds is not None and duration_seconds <= 5:
        return False
    return meaningful_word_count(text) < min_meaningful_words


def is_repetitive_hallucination(text: str) -> bool:
    compact_text = re.sub(r"\s+", " ", text).strip()
    if not compact_text:
        return True

    if re.fullmatch(r"[\W_]+", compact_text, flags=re.UNICODE):
        return True

    words = meaningful_words(compact_text)
    if not words:
        return True

    if any(normalize_for_match(phrase) in normalize_for_match(text) for phrase in GENERIC_HALLUCINATION_PHRASES):
        return True

    if len(words) < 20 and len(set(words)) <= 2:
        return True

    dominance = repeated_phrase_dominance(words)
    if dominance > 0.35:
        return True
    if len(words) < 40 and dominance > 0.25:
        return True
    if len(words) >= 40 and repeated_phrase_run(words) >= 4:
        return True
    return False


def repeated_phrase_dominance(words: list[str]) -> float:
    if not words:
        return 0.0
    max_repeated_words = 0
    for phrase_length in (2, 3, 4, 5):
        if len(words) < phrase_length * 3:
            continue
        counts: dict[str, int] = {}
        for index in range(0, len(words) - phrase_length + 1):
            phrase = " ".join(words[index : index + phrase_length])
            counts[phrase] = counts.get(phrase, 0) + 1
        if counts:
            max_count = max(counts.values())
            if max_count >= 3:
                max_repeated_words = max(max_repeated_words, max_count * phrase_length)
    return max_repeated_words / len(words)


def repeated_phrase_run(words: list[str]) -> int:
    max_run = 0
    for phrase_length in (2, 3, 4, 5):
        run = 1
        previous: tuple[str, ...] | None = None
        for index in range(0, len(words) - phrase_length + 1, phrase_length):
            phrase = tuple(words[index : index + phrase_length])
            if phrase == previous:
                run += 1
            else:
                run = 1
                previous = phrase
            max_run = max(max_run, run)
    return max_run


def is_soft_repetition_warning(
    text: str,
    duration_seconds: float | None,
    suspicious_signals: list[str],
    protective_signals: list[str],
) -> bool:
    if duration_seconds is None or duration_seconds <= 60:
        return False
    hard_signals = {
        "prompt_leak_detected",
        "segment_duration_exceeds_audio",
        "transcript_audio_length_mismatch",
        "generic_hallucination_phrase",
        "low_confidence_segments",
    }
    if any(signal in hard_signals for signal in suspicious_signals):
        return False
    return (
        meaningful_word_count(text) >= 40
        and ("contains_arabic_text" in protective_signals or "contains_latin_words" in protective_signals)
        and "sane_timestamps" in protective_signals
        and "no_prompt_leak" in protective_signals
    )


def is_short_generic_hallucination(
    text: str,
    duration_seconds: float | None,
    language_probability: float | None,
) -> bool:
    if duration_seconds is None or duration_seconds <= 5:
        return False

    words = meaningful_words(text)
    if len(words) > 9:
        return False

    normalized = normalize_for_match(text)
    if not normalized:
        return True

    trailing_connectors = {"a", "an", "the", "to", "of", "for", "with", "and", "or", "but"}
    if words and words[-1] in trailing_connectors and len(words) <= 8:
        return True

    if duration_seconds > 8 and len(words) <= 5 and not re.search(r"[.!?\u061f]$", text.strip()):
        return True

    weak_language = language_probability is None or language_probability < 0.65
    generic_starts = (
        "we will",
        "we are going",
        "in this video",
        "welcome to",
        "today we",
        "let us",
        "lets",
        "this video",
    )
    if weak_language and len(words) <= 8 and any(normalized.startswith(prefix) for prefix in generic_starts):
        return True

    return False


def meaningful_word_count(text: str) -> int:
    return len(meaningful_words(text))


def contains_arabic_text(text: str) -> bool:
    return any("\u0600" <= char <= "\u06ff" for char in text)


def contains_latin_words(text: str) -> bool:
    return any(re.search(r"[a-z]", word, flags=re.IGNORECASE) for word in meaningful_words(text))


def meaningful_words(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[\w\u0600-\u06FF]+", normalize_for_match(text), flags=re.UNICODE)
        if re.search(r"[\w\u0600-\u06FF]", word, flags=re.UNICODE)
    ]


def normalize_for_match(text: str) -> str:
    text = (text or "").casefold()
    text = re.sub(r"[^\w\u0600-\u06FF]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()
