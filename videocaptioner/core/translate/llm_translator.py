"""LLM 翻译器（使用 OpenAI）"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import json_repair
import openai

from videocaptioner.core.llm import call_llm
from videocaptioner.core.prompts import get_prompt
from videocaptioner.core.translate.base import BaseTranslator, SubtitleProcessData, logger
from videocaptioner.core.translate.types import TRANSLATION_REVIEW_MARKER, TargetLanguage
from videocaptioner.core.utils.cache import generate_cache_key

ENGLISH_WORD = r"[A-Za-z]+(?:['’][A-Za-z]+)?"
# Product names containing common English words, and identifiers such as A19 or 4K.
ENGLISH_PRODUCT_TERM = re.compile(
    r"(?<![A-Za-z0-9])(?:Find[ \t]+My(?:[ \t]+Remote|(?![ \t]+remote\b))|"
    r"(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*[0-9])[A-Z0-9]+)(?![A-Za-z0-9])"
)
COMMON_ENGLISH = {
    "a", "am", "and", "are", "been", "but", "can", "could", "do", "don't", "for",
    "ever", "have", "here", "how", "i", "in", "is", "it", "just", "me", "my", "never",
    "not", "of", "on", "or", "our", "so", "still", "that", "the", "their", "these",
    "they", "this", "to", "us", "wait", "was", "we", "were", "what", "when",
    "where", "will", "with", "would", "yeah", "you", "your",
}
NUMERIC_TIME = re.compile(r"(?<![\w:])(?:[01]?\d|2[0-3]):[0-5]\d(?![\d:])", re.ASCII)
CHINESE_TIME_PERIOD = re.compile(r"上午|下午|凌晨|早上|晚上")
ENGLISH_TIME_PERIOD = re.compile(
    r"\b(?:a\.?m\.?|p\.?m\.?|morning|afternoon|evening|night)\b", re.I
)
KANA_CHARACTERS = r"[\u3041-\u3096\u30a1-\u30fa\uff66-\uff6f\uff71-\uff9d]"
JAPANESE_KANA = re.compile(KANA_CHARACTERS)
KANA_LINE_BREAK = re.compile(
    rf"(?<={KANA_CHARACTERS})[^\S\r\n]*\r?\n[^\S\r\n]*(?={KANA_CHARACTERS})"
)


HIRAGANA = re.compile(r"[\u3041-\u309f]")
KATAKANA_RUN = re.compile(r"[\u30a1-\u30faー・\uff66-\uff9f]+")
HAN_CHARACTER = re.compile(r"[\u3400-\u9fff]")
LETTER_PAIR = re.compile(
    r"(?<![A-Za-z])([A-Z])\s*(?:とか|か|や|或是|或者|或|與|和|跟|、|[/／])\s*"
    r"([A-Z])(?![A-Za-z])"
)


def copied_english(
    source: str, translation: str, *, current_text: Optional[str] = None
) -> list[str]:
    """Flag copied prose while allowing names, acronyms and paired letter labels."""
    current_text = current_text if current_text is not None else source
    source_terms = {
        " ".join(term.split()) for term in ENGLISH_PRODUCT_TERM.findall(current_text)
    }
    translation = ENGLISH_PRODUCT_TERM.sub(
        lambda match: "，" if " ".join(match.group().split()) in source_terms else match.group(),
        translation,
    )
    source_words = set(re.findall(ENGLISH_WORD, source.casefold()))
    source_pairs = {frozenset(pair) for pair in LETTER_PAIR.findall(current_text)}
    translation = LETTER_PAIR.sub(
        lambda match: " " if frozenset(match.groups()) in source_pairs else match.group(),
        translation,
    )
    suspicious = []
    for phrase in re.findall(rf"{ENGLISH_WORD}(?:[ \t]+{ENGLISH_WORD})*", translation):
        words = phrase.split()
        if (
            phrase in source
            and all(word[:1].isupper() for word in words)
            and not any(
                word.casefold() in COMMON_ENGLISH and not (len(word) > 1 and word.isupper())
                for word in words
            )
        ):
            continue
        copied = [word for word in words if word.casefold() in source_words]
        if sum(word.islower() for word in copied) >= 2 or any(
            word.casefold() in COMMON_ENGLISH for word in copied
        ):
            suspicious.append(phrase)
    return suspicious


def mark_untranslated_english(source: str, translation: str, phrases: list[str]) -> Optional[str]:
    """Mark copied text only when the entire fragment comes from the current subtitle."""
    patterns = [
        r"(?<![A-Za-z0-9])" + r"[ \t]+".join(map(re.escape, phrase.split())) + r"(?![A-Za-z0-9])"
        for phrase in phrases
    ]
    if any(not re.search(pattern, source, re.IGNORECASE) for pattern in patterns):
        return None
    for pattern in patterns:
        translation = re.sub(pattern, TRANSLATION_REVIEW_MARKER, translation)
    return translation


def mark_unresolved_katakana(source: str, translation: str) -> Optional[str]:
    """Mask intact, copied katakana terms while rejecting untranslated Japanese prose."""
    source_runs = set(KATAKANA_RUN.findall(source))
    remaining_runs = KATAKANA_RUN.findall(translation)
    if HIRAGANA.search(translation) or not remaining_runs or any(
        run not in source_runs for run in remaining_runs
    ):
        return None
    # Reject verbatim Japanese text with only punctuation or layout changes.
    if re.sub(r"\W", "", source) == re.sub(r"\W", "", translation):
        return None
    return KATAKANA_RUN.sub(TRANSLATION_REVIEW_MARKER, translation)


class LLMTranslator(BaseTranslator):
    """LLM 翻译器（OpenAI兼容API）"""

    MAX_STEPS = 3

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        model: str,
        custom_prompt: str,
        is_reflect: bool,
        update_callback: Optional[Callable],
    ):
        super().__init__(
            thread_num=thread_num,
            batch_num=batch_num,
            target_language=target_language,
            update_callback=update_callback,
        )

        self.model = model
        self.custom_prompt = custom_prompt
        self.is_reflect = is_reflect

    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        logger.debug(
            f"[+]正在翻译字幕: {subtitle_chunk[0].index} - {subtitle_chunk[-1].index}"
        )
        if (
            len(subtitle_chunk) == 1
            and not self.is_reflect
            and self.target_language is TargetLanguage.TRADITIONAL_CHINESE
        ):
            return self._translate_traditional_cue(subtitle_chunk)

        # 转换为字典格式用于API调用
        subtitle_dict = {
            str(data.index): {
                "previous_context": data.previous_context,
                "current_subtitle": data.original_text,
                "next_context": data.next_context,
            }
            for data in subtitle_chunk
        }
        if len(subtitle_chunk) == 1 and not self.is_reflect:
            data = subtitle_chunk[0]
            subtitle_dict = {str(data.index): {"current_subtitle": data.original_text}}

        # 获取提示词
        if self.is_reflect:
            prompt = get_prompt(
                "translate/reflect",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )
        else:
            prompt = get_prompt(
                "translate/standard",
                target_language=self.target_language,
                custom_prompt=self.custom_prompt,
            )
        prompt += (
            "\nThe previous_context and next_context fields are reference only. "
            "Translate ONLY current_subtitle. Never include, paraphrase, or return "
            "the context text. Return one translated string for each input key."
        )

        try:
            # 使用agent loop进行翻译，自动验证和修正
            result_dict = self._agent_loop(prompt, subtitle_dict)

            # 处理反思翻译模式的结果
            if self.is_reflect and isinstance(result_dict, dict):
                processed_result = {
                    k: f"{v.get('native_translation', v) if isinstance(v, dict) else v}"
                    for k, v in result_dict.items()
                }
            else:
                processed_result = {k: f"{v}" for k, v in result_dict.items()}

            # 将结果填充回SubtitleProcessData
            for data in subtitle_chunk:
                data.translated_text = processed_result.get(
                    str(data.index), ""
                )
            return subtitle_chunk
        except openai.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {str(e)}")
            raise
        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {str(e)}")
            raise
        except openai.NotFoundError as e:
            logger.error(f"OpenAI NotFound Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"LLM translation error: {e}")
            raise

    def _translate_traditional_cue(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        cue = subtitle_chunk[0]
        is_japanese = JAPANESE_KANA.search(cue.original_text)
        source = KANA_LINE_BREAK.sub("", cue.original_text) if is_japanese else cue.original_text
        source_text = " ".join(source.splitlines())
        prompt = (
            "你是繁體中文字幕譯者。只翻譯使用者訊息末尾「本條字幕」的內容。"
            "參考字幕只供判斷語意和語氣，不譯入回覆。"
            "逐句對應時間軸，原文是半句時，譯文也只保留同一半句。"
            "忠實保留原文的主詞、受詞、否定、數字和人名。"
            "方括號內的 TIME_ 標記代表時刻，原樣保留，勿加上未提供的時段。"
            "表示未來的 if you ever 可譯為「如果你有一天」。"
            "逐一翻譯所有能理解的英文詞句，英文人名、品牌及縮寫可按需要保留。"
            "粵語內容保留原詞並使用繁體字。"
            "遇到疑似辨識錯字，只翻譯可確認的意思，不補寫歌詞或猜測缺失內容。"
            "輸出一行譯文純文字，不附解釋、JSON、引號或原文。"
            "例如本條是「And in the morning,」，下一條是「we become strangers.」，"
            "只輸出「而到了早晨，」。"
        )
        if self.custom_prompt:
            prompt += f"\n額外要求：{self.custom_prompt}"
        if is_japanese:
            prompt += (
                "\n本條是日文或中日混合字幕，請翻譯其中的日文為繁體中文。"
                "保留問句、否定與口語語氣，請勿用日文改寫。"
                "保留比較和最高級，例如「やっぱり自分の家が一番」譯為「果然還是自己家最好」。"
                "人名與專名使用常見中文譯名、漢字或羅馬字。"
                "像 I 或 F 這種表示分類的字母原樣保留。"
            )
        context = json.dumps(
            {
                "previous_context": cue.previous_context,
                "next_context": cue.next_context,
            },
            ensure_ascii=False,
        )
        time_markers = {}
        for match in NUMERIC_TIME.finditer(source_text):
            token = f"[TIME_{len(time_markers) + 1}]"
            while token in source_text or token in time_markers:
                token = f"[{token}]"
            time_markers[token] = match.group()
        if time_markers:
            markers = iter(time_markers)
            marked_source = NUMERIC_TIME.sub(lambda _: next(markers), source_text)
        else:
            marked_source = source_text
        time_period_unspecified = (
            bool(time_markers)
            and not ENGLISH_TIME_PERIOD.search(source_text)
            and not CHINESE_TIME_PERIOD.search(source_text)
        )
        user_message = f"參考字幕：{context}\n本條字幕：{marked_source}"
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": user_message}]
        review_candidate = None
        surrounding_text = f"{cue.previous_context} {cue.original_text} {cue.next_context}"
        for attempt in range(self.MAX_STEPS):
            self.check_cancelled()
            request = call_llm.__wrapped__ if self.force_retry else call_llm
            response = request(messages=messages, model=self.model, temperature=0.2 + attempt * 0.1)
            content = response.choices[0].message.content or ""
            translation = content.strip().strip('"`「」').strip()
            remaining = copied_english(surrounding_text, translation, current_text=source_text)
            remaining_japanese = JAPANESE_KANA.search(translation)
            invalid_markers = [token for token in time_markers if translation.count(token) != 1]
            added_period = time_period_unspecified and CHINESE_TIME_PERIOD.search(translation)
            candidate = None
            if (
                translation
                and not invalid_markers
                and not added_period
                and r"\n" not in translation
                and len(translation.splitlines()) == 1
            ):
                for token, value in time_markers.items():
                    translation = translation.replace(token, value)
                if not remaining and not remaining_japanese and TRANSLATION_REVIEW_MARKER not in translation:
                    cue.translated_text = translation
                    return subtitle_chunk
                if self.allow_partial:
                    if remaining and not remaining_japanese:
                        candidate = mark_untranslated_english(source_text, translation, remaining)
                    elif is_japanese and not remaining:
                        candidate = (
                            mark_unresolved_katakana(source_text, translation)
                            if remaining_japanese else translation
                        )
                    if candidate and not HAN_CHARACTER.search(
                        candidate.replace(TRANSLATION_REVIEW_MARKER, "")
                    ):
                        candidate = None
                    if candidate and review_candidate is None:
                        review_candidate = candidate
            retry_prompt = prompt + (
                "\n請重新翻譯。只翻譯本條的文字，保留意思和語氣，"
                "輸出一行可直接放進字幕的繁體中文。"
            )
            if remaining:
                retry_prompt += f"這些詞句也必須翻譯：{', '.join(remaining)}。"
                if candidate:
                    retry_prompt += "保留已譯出的意思，只修正未譯片段。"
            if remaining_japanese or TRANSLATION_REVIEW_MARKER in translation:
                retry_prompt += (
                    "\n上次回覆仍有日文假名或待核對片段。保留已譯出的意思，"
                    "只修正未譯片段，照原文保留問句、語氣和說話者，勿自行加上「我是」等原文沒有的內容。"
                    "假名若是一般日語，請完整翻譯。專名有可靠譯名時可使用漢字或羅馬字。"
                )
                if self.allow_partial:
                    retry_prompt += (
                        f"若疑似語音辨識有誤且無法判斷，僅將該片段寫為{TRANSLATION_REVIEW_MARKER}。"
                    )
            if invalid_markers:
                retry_prompt += f"原樣保留這些時刻標記，各一次：{', '.join(invalid_markers)}。"
            if added_period:
                retry_prompt += "原文未指明時段，請刪除自行加入的時段。"
            if any("still" in phrase.casefold().split() for phrase in remaining):
                retry_prompt += "表示延續時，將 still 譯為「還」、「依然」或「仍然」。"
            messages = [
                {"role": "system", "content": retry_prompt},
                {"role": "user", "content": user_message},
            ]
            if candidate and (remaining or remaining_japanese):
                messages.extend((
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": "請依指示只修正這一條譯文，直接輸出修正後的一行字幕。"},
                ))
        if review_candidate is not None:
            logger.info("字幕第 %s 條保留部分譯文，已標記待核對", cue.index)
            cue.translated_text = review_candidate
            return subtitle_chunk
        raise RuntimeError(f"字幕第 {cue.index} 條未通過譯文檢查，請核對原文或重新翻譯。")

    def _agent_loop(
        self, system_prompt: str, subtitle_dict: Dict[str, str]
    ) -> Dict[str, str]:
        """Agent loop翻译字幕块"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(subtitle_dict, ensure_ascii=False)},
        ]
        # llm 反馈循环
        for _ in range(self.MAX_STEPS):
            self.check_cancelled()
            request = call_llm.__wrapped__ if self.force_retry else call_llm
            response = request(messages=messages, model=self.model)
            response_dict = json_repair.loads(
                response.choices[0].message.content.strip()
            )
            is_valid, error_message = self._validate_llm_response(
                response_dict, subtitle_dict
            )
            if is_valid:
                return response_dict
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(response_dict, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"Error: {error_message}\n\nFix the errors above and output ONLY a valid JSON dictionary with ALL {len(subtitle_dict)} keys",
                    }
                )

        raise RuntimeError(f"字幕翻譯回覆格式不符：{error_message}")

    def _validate_llm_response(
        self, response_dict: Any, subtitle_dict: Dict[str, str]
    ) -> Tuple[bool, str]:
        """验证LLM翻译结果（支持普通和反思模式）

        Returns: (is_valid, error_feedback)
        """
        if not isinstance(response_dict, dict):
            return (
                False,
                f"Output must be a dict, got {type(response_dict).__name__}. Use format: {{'0': 'text', '1': 'text'}}",
            )

        expected_keys = set(subtitle_dict.keys())
        actual_keys = set(response_dict.keys())

        def sort_keys(keys):
            return sorted(keys, key=lambda x: int(x) if x.isdigit() else x)

        # 检查键是否匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            error_parts = []

            if missing:
                error_parts.append(
                    f"Missing keys {sort_keys(missing)} - you must translate these items"
                )
            if extra:
                error_parts.append(
                    f"Extra keys {sort_keys(extra)} - these keys are not in input, remove them"
                )

            return (False, "; ".join(error_parts))

        # 如果是反思模式，检查嵌套结构
        if self.is_reflect:
            for key, value in response_dict.items():
                if not isinstance(value, dict):
                    return (
                        False,
                        f"Key '{key}': value must be a dict with 'native_translation' field. Got {type(value).__name__}.",
                    )

                if "native_translation" not in value:
                    available_keys = list(value.keys())
                    return (
                        False,
                        f"Key '{key}': missing 'native_translation' field. Found keys: {available_keys}. Must include 'native_translation'.",
                    )

        if self.target_language is TargetLanguage.TRADITIONAL_CHINESE:
            for key, value in response_dict.items():
                translation = value["native_translation"] if self.is_reflect else value
                if not isinstance(translation, str) or not translation.strip():
                    return False, f"Key '{key}': return a non-empty Traditional Chinese translation."
                if JAPANESE_KANA.search(translation):
                    return False, f"Key '{key}': translate the remaining Japanese into Traditional Chinese."

        return True, ""

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        model = self.model
        if self.target_language is TargetLanguage.TRADITIONAL_CHINESE:
            suffix = f":hant-v11:{self.is_reflect}:{generate_cache_key(self.custom_prompt)}"
        else:
            suffix = ":single-v2" if len(chunk) == 1 and not self.is_reflect else ""
        return f"{class_name}:{chunk_key}:{lang}:{model}{suffix}"
