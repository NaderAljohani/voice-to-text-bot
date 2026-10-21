import asyncio
import shutil
import wave
import sys

import pytest

import audio_utils


@pytest.mark.parametrize('raw,expected', [('N/A\n1.2\n1.4', 2), ('12.0', 12)])
def test_duration_rounds_up(monkeypatch, tmp_path, raw, expected):
    async def fake(*args):
        return raw
    monkeypatch.setattr(audio_utils, '_run_command', fake)
    assert asyncio.run(audio_utils.get_audio_duration_seconds(tmp_path / 'input')) == expected


@pytest.mark.parametrize('raw', ['N/A', '0', '-1', 'nan', 'inf'])
def test_invalid_duration_rejected(monkeypatch, tmp_path, raw):
    async def fake(*args):
        return raw
    monkeypatch.setattr(audio_utils, '_run_command', fake)
    with pytest.raises(RuntimeError):
        asyncio.run(audio_utils.get_audio_duration_seconds(tmp_path / 'input'))


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg not on PATH')
def test_real_ffmpeg_conversion_and_silence_vad(tmp_path):
    source, target = tmp_path / 'source.wav', tmp_path / 'converted.wav'
    with wave.open(str(source), 'wb') as stream:
        stream.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\0\0\0\0' * 44100)
    asyncio.run(audio_utils.convert_to_wav_16k_mono(source, target))
    with wave.open(str(target)) as stream:
        assert (stream.getnchannels(), stream.getframerate(), stream.getsampwidth()) == (1, 16000, 2)
    assert asyncio.run(audio_utils.has_audio_stream(target))
    assert asyncio.run(audio_utils.get_audio_duration_seconds(target)) == 1
    vad = audio_utils.detect_speech_in_wav(target)
    assert vad.available
    assert vad.detected_speech_seconds == 0


@pytest.mark.parametrize('cancel', [False, True])
def test_subprocess_terminated_on_timeout_or_cancellation(monkeypatch, cancel):
    original = asyncio.create_subprocess_exec
    processes = []
    async def run():
        started = asyncio.Event()
        async def capture(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            started.set()
            return process
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', capture)
        task = asyncio.create_task(audio_utils._run_command(
            sys.executable, '-c', 'import time; time.sleep(30)',
            timeout_seconds=30 if cancel else 0.1))
        if cancel:
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match='timed out'):
                await task
        assert processes[0].returncode is not None
    asyncio.run(run())
