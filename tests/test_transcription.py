import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import transcriber
from transcript_qc import TranscriptQcSettings, validate_transcription
from transcriber import (
    LocalWhisperTranscriber, TranscriptionResult, TranscriptionSegment,
    format_timestamped_transcript, split_wav_into_chunks, write_wav_chunk,
)


def make_result(text, end=5, **metrics):
    segments = [TranscriptionSegment(0, end, text, **metrics)] if text else []
    return TranscriptionResult(text, format_timestamped_transcript(segments), segments, 'ar', 0.95)


@pytest.mark.parametrize('text,end,reason', [
    ('', 5, 'empty_transcript'),
    ('Transcribe exactly what is spoken', 5, 'prompt_leak_detected'),
    ('هذا نص لاختبار حدود المقطع', 20, 'segment_duration_exceeds_audio'),
])
def test_qc_rejects_invalid_results(text, end, reason):
    result = validate_transcription(make_result(text, end), duration_seconds=5, settings=TranscriptQcSettings())
    assert not result.passed
    assert result.reason == reason


def test_qc_accepts_mixed_language_speech():
    result = validate_transcription(make_result('نراجع اليوم خطة المشروع ونستخدم Python لتنظيف البيانات'),
                                    duration_seconds=5, settings=TranscriptQcSettings())
    assert result.passed
    assert 'Python' in result.result.text


def test_qc_rejects_weak_segments():
    result = validate_transcription(make_result('هذا نص تجريبي واضح', no_speech_prob=0.99, avg_logprob=-4),
                                    duration_seconds=5, settings=TranscriptQcSettings())
    assert not result.passed


def test_qc_can_be_disabled():
    result = validate_transcription(make_result(''), duration_seconds=5, settings=TranscriptQcSettings(enabled=False))
    assert result.passed
    assert 'qc_skipped' in result.protective_signals


@pytest.mark.parametrize('seconds', [1, 20, 21, 40, 61])
def test_chunks_cover_audio_with_valid_sample_ranges(tmp_path, seconds):
    audio = tmp_path / 'generated.wav'
    with wave.open(str(audio), 'wb') as stream:
        stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0' * 16000 * seconds)
    chunks = split_wav_into_chunks(audio, chunk_seconds=20, overlap_seconds=3, min_chunk_seconds=8)
    assert chunks[0].start_sample == 0
    assert chunks[-1].end_sample == 16000 * seconds
    for index, chunk in enumerate(chunks):
        assert chunk.end_sample > chunk.start_sample
        if index:
            assert chunk.start_sample <= chunks[index-1].end_sample
    output = tmp_path / 'chunk.wav'
    write_wav_chunk(audio, output, chunks[0].start_sample, chunks[0].end_sample)
    with wave.open(str(output)) as stream:
        assert stream.getnframes() == chunks[0].end_sample - chunks[0].start_sample


def test_inference_serialized_and_model_loaded_once(monkeypatch, tmp_path):
    active = 0
    peak = 0
    loaded = []
    guard = threading.Lock()
    options = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            loaded.append(1)

        def transcribe(self, path, **kwargs):
            nonlocal active, peak
            options.append(kwargs)
            def segments():
                nonlocal active, peak
                with guard:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                yield SimpleNamespace(start=0, end=1, text='test words')
                with guard:
                    active -= 1
            return segments(), SimpleNamespace(language='en', language_probability=0.99)

    monkeypatch.setattr(transcriber, 'WhisperModel', FakeModel)
    model = LocalWhisperTranscriber('small', 'cpu', 'int8', initial_prompt='test prompt')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: model.transcribe_normal(tmp_path / 'synthetic.wav', use_initial_prompt=False), range(4)))
    assert len(loaded) == 1
    assert peak == 1
    assert all(result.text == 'test words' for result in results)
    assert all('initial_prompt' not in option and option['task'] == 'transcribe' for option in options)
