import asyncio
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import bot
from tests.test_transcription import make_result


@pytest.fixture
def job(monkeypatch, tmp_path):
    result = make_result('نراجع اليوم خطة المشروع ونستخدم Python لتنظيف البيانات')
    message = SimpleNamespace(from_user=SimpleNamespace(id=42), message_id=1,
                              voice=None, audio=None, video=None, video_note=None, document=None,
                              answer=AsyncMock(), answer_document=AsyncMock())
    transport = SimpleNamespace(get_file=AsyncMock(return_value=SimpleNamespace(file_path='synthetic')),
                                download_file=AsyncMock())
    def download(path, destination):
        destination.write_bytes(b'synthetic')
    transport.download_file.side_effect = download
    async def convert(source, target):
        assert source != target, 'WAV uploads must have a distinct conversion output'
        target.write_bytes(b'converted')
    monkeypatch.setattr(bot, 'reject_if_banned', AsyncMock(return_value=False))
    monkeypatch.setattr(bot, 'register_user', AsyncMock())
    monkeypatch.setattr(bot, 'get_media', lambda _: SimpleNamespace(file_id='synthetic'))
    monkeypatch.setattr(bot, 'is_supported_media', lambda _: True)
    monkeypatch.setattr(bot, 'media_suffix', lambda _: '.wav')
    monkeypatch.setattr(bot, 'save_transcription_log', Mock())
    monkeypatch.setattr(bot, 'DOWNLOADS_DIR', tmp_path)
    monkeypatch.setattr(bot, 'TRANSCRIPTS_DIR', tmp_path)
    monkeypatch.setattr(bot, 'speech_vad_enabled', False)
    monkeypatch.setattr(bot, 'store_audio_files', False)
    monkeypatch.setattr(bot, 'admin_telegram_ids', frozenset())
    monkeypatch.setattr(bot, 'user_locks', defaultdict(asyncio.Lock))
    monkeypatch.setattr(bot, 'get_audio_duration_seconds', AsyncMock(return_value=5))
    monkeypatch.setattr(bot, 'convert_to_wav_16k_mono', convert)
    monkeypatch.setattr(bot, 'should_run_chunk_fallback', lambda *args: False)
    monkeypatch.setattr(bot, 'balance_text_for_user', lambda _: 'balance')
    database = Mock()
    database.balance_info.return_value = {'total_available_seconds': 600}
    monkeypatch.setattr(bot, 'db', database, raising=False)
    model = SimpleNamespace(initial_prompt=None, transcribe_normal=Mock(return_value=result))
    monkeypatch.setattr(bot, 'transcriber', model, raising=False)
    return message, transport, database, model, tmp_path


def test_wav_success_delivers_before_deducting_and_cleans_audio(job):
    message, transport, database, model, directory = job
    def deduct(*args):
        assert message.answer_document.await_count == 2
    database.deduct_seconds.side_effect = deduct
    asyncio.run(bot.handle_audio(message, transport))
    assert message.answer_document.await_count == 2
    database.deduct_seconds.assert_called_once()
    assert not list(directory.glob('*.wav'))
    assert len(list(directory.glob('*.txt'))) == 2


def test_delivery_failure_does_not_charge(job):
    message, transport, database, model, directory = job
    message.answer_document.side_effect = RuntimeError('synthetic send failure')
    asyncio.run(bot.handle_audio(message, transport))
    model.transcribe_normal.assert_called_once()
    message.answer_document.assert_awaited_once()
    database.deduct_seconds.assert_not_called()
    assert not list(directory.glob('*.wav'))


def test_no_balance_skips_inference(job):
    message, transport, database, model, _ = job
    database.balance_info.return_value = {'total_available_seconds': 0}
    asyncio.run(bot.handle_audio(message, transport))
    model.transcribe_normal.assert_not_called()
    database.deduct_seconds.assert_not_called()


def test_invalid_transcript_retries_without_prompt_and_never_charges(job):
    message, transport, database, model, _ = job
    model.transcribe_normal.return_value = make_result('')
    asyncio.run(bot.handle_audio(message, transport))
    assert model.transcribe_normal.call_count == 2
    assert model.transcribe_normal.call_args.kwargs['use_initial_prompt'] is False
    database.deduct_seconds.assert_not_called()
    message.answer_document.assert_not_awaited()


def test_same_user_second_request_is_rejected_while_first_waits(job):
    message, transport, database, model, _ = job
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def get_file(*args):
            entered.set()
            await release.wait()
            return SimpleNamespace(file_path='synthetic')
        transport.get_file.side_effect = get_file
        first = asyncio.create_task(bot.handle_audio(message, transport))
        await entered.wait()
        await bot.handle_audio(message, transport)
        release.set()
        await first
    asyncio.run(run())
    transport.get_file.assert_awaited_once()
    assert any(call.args[0] == bot.USER_ALREADY_PROCESSING for call in message.answer.call_args_list)
