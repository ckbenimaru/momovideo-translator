from threading import get_ident
from types import SimpleNamespace

import pytest
from diskcache import Cache

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.entities import SubtitleProcessData
from videocaptioner.core.translate.llm_translator import LLMTranslator, copied_english
from videocaptioner.core.translate.types import TRANSLATION_REVIEW_MARKER, TargetLanguage


def test_copied_english_flags_leakage_and_keeps_names():
    assert copied_english("These still last so", "這些 still last so")
    assert copied_english("Still remember when we met", "still 記得相遇的時候")
    assert copied_english("You just wait", "你 just wait")
    assert copied_english("Oh yeah, oh yeah", "哦 Yeah，哦 Yeah")
    assert copied_english("So if you ever", "所以如果你 ever")
    assert copied_english("Still remember when we met", "Still 記得相遇時")
    assert not copied_english("I saw IBM and Sam in Hong Kong", "我在 Hong Kong 見到 Sam 和 IBM")


def test_single_traditional_translation_retries_context_leak(monkeypatch):
    calls = []
    replies = iter(["我們 still 有靈魂\n只有上帝能將我們分開", "只有上帝能將我們分開"])

    def fake_call(**kwargs):
        calls.append(kwargs["messages"])
        message = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    cue = SubtitleProcessData(
        7,
        "and only God could separate us.",
        previous_context="we still have souls,",
        next_context="On my skin you left your mark.",
    )
    try:
        result = translator._translate_chunk([cue])
    finally:
        translator.stop()
    assert result[0].translated_text == "只有上帝能將我們分開"
    assert '"previous_context": "we still have souls,"' in calls[0][1]["content"]
    assert calls[0][1]["content"].endswith(cue.original_text)
    assert calls[1][1]["content"] == calls[0][1]["content"]
    assert "仍然" in calls[1][0]["content"]


def test_parallel_translation_reports_fresh_and_cached_cues_on_task_thread(
    monkeypatch, tmp_path
):
    worker_calls = []
    updates = []
    task_thread = get_ident()

    def fake_translate(chunk):
        worker_calls.append(get_ident())
        chunk[0].translated_text = f"字幕 {chunk[0].index}"
        return chunk

    def callback(chunk):
        updates.append((get_ident(), chunk[0].index))

    translator = LLMTranslator(4, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, callback)
    monkeypatch.setattr(translator, "_translate_chunk", fake_translate)
    with Cache(str(tmp_path / "translations")) as cache:
        translator._cache = cache
        try:
            for _ in range(2):
                data = ASRData([
                    ASRDataSeg(text=f"Line {i}", start_time=i * 1000, end_time=(i + 1) * 1000)
                    for i in range(4)
                ])
                result = translator.translate_subtitle(data)
                assert [seg.translated_text for seg in result.segments] == [
                    "字幕 1", "字幕 2", "字幕 3", "字幕 4",
                ]
        finally:
            translator.stop()
    assert len(worker_calls) == 4
    assert all(worker != task_thread for worker in worker_calls)
    assert sorted(updates) == [
        (task_thread, index) for index in range(1, 5) for _ in range(2)
    ]


def test_single_traditional_translation_preserves_numeric_time(monkeypatch):
    replies = iter(["團隊要到明天下午[TIME_1]才開會。", "團隊要到明天[TIME_1]才開會。"])

    def fake_call(**kwargs):
        assert "until [TIME_1] tomorrow" in kwargs["messages"][1]["content"]
        message = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    cue = SubtitleProcessData(2, "The team won't meet until 2:00 tomorrow.")
    try:
        result = translator._translate_chunk([cue])
    finally:
        translator.stop()
    assert result[0].translated_text == "團隊要到明天2:00才開會。"


def test_traditional_translation_retries_japanese_paraphrase(monkeypatch):
    replies = iter(["やっぱり好きな人が一番。", "果然還是和喜歡的人一起最好。"])
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["messages"])
        message = SimpleNamespace(content=next(replies))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    cue = SubtitleProcessData(119, "やっぱり好きな人が一番だと思いま\nす。")
    try:
        assert translator._translate_chunk([cue])[0].translated_text == "果然還是和喜歡的人一起最好。"
        assert not translator._validate_llm_response({"119": "這樣が最好"}, {"119": {}})[0]
    finally:
        translator.stop()
    assert "中日混合字幕" in calls[0][0]["content"]
    assert "日文假名" in calls[1][0]["content"]
    assert calls[0][1]["content"].endswith("だと思います。")
    assert calls[0][1] == calls[1][1]


def test_copied_english_keeps_matching_letter_labels_in_place():
    source = "I was asking. 型はIとかFとか"
    assert not copied_english(source, "我想問是 I 或 F。")
    assert copied_english(source, "I was asking，是 I 或 F。")
    assert copied_english(source, "是 I 或 F。", current_text="I was asking.")
    assert copied_english(source, "是 I 或 G。")


@pytest.mark.parametrize(
    ("source", "replies", "expected"),
    [
        (
            "こんばんはすみませんテストネ。 うん、なんですか？",
            [
                "晚上好，不好意思，テストネ。嗯，什麼事？",
                "晚上好，不好意思，我是テストネ。嗯，什麼事？",
                "晚上好，不好意思，我是テストネ。嗯，什麼事？",
            ],
            f"晚上好，不好意思，{TRANSLATION_REVIEW_MARKER}。嗯，什麼事？",
        ),
        (
            "あ、フニフニ",
            ["啊、フニフニ"] * 3,
            f"啊、{TRANSLATION_REVIEW_MARKER}",
        ),
        (
            "製品はIとかFとか、そうですね。 はい",
            ["產品型號是 I 或 F，對。"],
            "產品型號是 I 或 F，對。",
        ),
        (
            "こんばんはすみませんテストネ。 うん、なんですか？",
            ["晚上好，不好意思，テストネ。嗯，什麼事？", "晚上好，不好意思。嗯，什麼事？"],
            "晚上好，不好意思。嗯，什麼事？",
        ),
    ],
)
def test_retranslate_preserves_usable_text(
    monkeypatch, tmp_path, source, replies, expected
):
    calls = []
    responses = iter(replies)

    def fake_call(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
        )

    def cached_call(**kwargs):
        pytest.fail("Explicit retry should bypass the response cache")

    cached_call.__wrapped__ = fake_call
    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", cached_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    with Cache(str(tmp_path / "retry-cache")) as cache:
        translator._cache = cache
        try:
            result = translator.translate_subtitle(
                ASRData([ASRDataSeg(text=source, start_time=0, end_time=1000)]),
                allow_partial=True,
                force_retry=True,
            )
            assert result.segments[0].translated_text == expected
            assert len(cache) == int(TRANSLATION_REVIEW_MARKER not in expected)
        finally:
            translator.stop()
    assert len(calls) == len(replies)
    if len(calls) > 1:
        assert TRANSLATION_REVIEW_MARKER in calls[1][0]["content"]
        assert calls[1][2] == {"role": "assistant", "content": replies[0]}


@pytest.mark.parametrize(
    ("source", "reply", "allow_partial"),
    [
        ("やっぱり好きな人が一番。", "やっぱり好きな人が一番。", True),
        ("テストネ", "テストネ", True),
        ("こんばんはテストネ。", "晚上好，テストネ。", False),
        ("こんばんはテストネ。", "晚上好，サトウ。", True),
        ("こんばんはテストネ。", "［待確認］", True),
        ("テストネに2:00に会いましょう。", "和テストネ見面。", True),
    ],
)
def test_review_fallback_requires_usable_text_and_valid_time(
    monkeypatch, source, reply, allow_partial
):
    def fake_call(**kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    translator.allow_partial = allow_partial
    try:
        with pytest.raises(RuntimeError, match="未通過譯文檢查"):
            translator._translate_chunk([SubtitleProcessData(1, source)])
    finally:
        translator.stop()


@pytest.mark.parametrize(
    ("source", "reply"),
    [
        (
            "The Siri remote also works with Find My, so if you happen to lose your remote, guess what?",
            "Siri 遠端控制器也支援 Find My，所以如果你有一天遺失了遠端控制器，猜猜看？",
        ),
        (
            "Siri can help you find it. We're going to do a Find My Remote demo.",
            "Siri 可以幫你找到它。我們要來做一個 Find My Remote 示範。",
        ),
        (
            "So here's the new Apple TV 4K, and this runs on the A19 chip",
            "所以這就是全新的 Apple TV 4K，它搭載 A19 晶片。",
        ),
    ],
)
def test_traditional_translation_accepts_logged_apple_product_names(monkeypatch, source, reply):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    try:
        cue = SubtitleProcessData(1, source)
        assert translator._translate_chunk([cue])[0].translated_text == reply
        assert len(calls) == 1
    finally:
        translator.stop()


def test_product_names_do_not_exempt_english_prose_or_context_only_terms():
    assert copied_english("Use Find My and wait here", "使用 Find My，and wait here")
    assert copied_english("Find My remote, please", "請幫我 Find My remote")
    assert copied_english("Find My. Set it up.", "設定 Find My", current_text="Set it up.")
    assert copied_english("A19 and a remote", "有一個 A 遙控器")


@pytest.mark.parametrize(
    ("source", "reply", "expected"),
    [
        (
            "The remote still works",
            "遙控器 still 可以用",
            f"遙控器 {TRANSLATION_REVIEW_MARKER} 可以用",
        ),
        ("The remote works", "遙控器 still 可以用", None),
        ("Still works", "Still works", None),
        ("The team waits until 2:00", "團隊 still 要等到下午[TIME_1]", None),
    ],
)
def test_review_fallback_marks_only_english_from_current_cue(monkeypatch, source, reply, expected):
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs["messages"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    monkeypatch.setattr("videocaptioner.core.translate.llm_translator.call_llm", fake_call)
    translator = LLMTranslator(1, 1, TargetLanguage.TRADITIONAL_CHINESE, "local", "", False, None)
    translator.allow_partial = True
    cue = SubtitleProcessData(1, source, previous_context="Still remember?")
    try:
        if expected is None:
            with pytest.raises(RuntimeError, match="未通過譯文檢查"):
                translator._translate_chunk([cue])
        else:
            assert translator._translate_chunk([cue])[0].translated_text == expected
            assert calls[1][2] == {"role": "assistant", "content": reply}
        assert len(calls) == translator.MAX_STEPS
    finally:
        translator.stop()
