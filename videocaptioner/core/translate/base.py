"""翻译器基类"""

import atexit
import time
from abc import ABC, abstractmethod
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import replace
from threading import Event
from typing import Callable, List, Optional

from videocaptioner.core.asr.asr_data import ASRData, ASRDataSeg
from videocaptioner.core.entities import SubtitleProcessData
from videocaptioner.core.translate.types import TargetLanguage, translation_needs_review
from videocaptioner.core.utils.cache import generate_cache_key, get_translate_cache
from videocaptioner.core.utils.chinese_script import normalize_translated_script
from videocaptioner.core.utils.logger import setup_logger

logger = setup_logger("subtitle_translator")


class BaseTranslator(ABC):
    """翻译器基类"""

    def __init__(
        self,
        thread_num: int,
        batch_num: int,
        target_language: TargetLanguage,
        update_callback: Optional[Callable],
    ):
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.target_language = target_language
        self.is_running = True
        self.update_callback = update_callback
        self.executor = None
        self._cache = get_translate_cache()
        self.cancel_event: Optional[Event] = None
        self.force_retry = False
        self.allow_partial = False

        self._init_thread_pool()

    def _init_thread_pool(self):
        """初始化线程池"""
        self.executor = ThreadPoolExecutor(max_workers=self.thread_num)
        atexit.register(self.stop)

    def translate_subtitle(
        self,
        subtitle_data: ASRData,
        *,
        context_by_index: Optional[dict[int, tuple[str, str]]] = None,
        cancel_event: Optional[Event] = None,
        allow_partial: bool = False,
        force_retry: bool = False,
    ) -> ASRData:
        """翻译字幕文件"""
        try:
            self.cancel_event = cancel_event
            self.force_retry = force_retry
            self.allow_partial = allow_partial
            self.check_cancelled()
            asr_data = subtitle_data

            # 将ASRData转换为SubtitleProcessData列表
            translate_data_list = []
            for i, seg in enumerate(asr_data.segments):
                previous = asr_data.segments[i - 1].text if i > 0 else ""
                following = asr_data.segments[i + 1].text if i + 1 < len(asr_data.segments) else ""
                if context_by_index:
                    previous, following = context_by_index.get(i + 1, (previous, following))
                translate_data_list.append(
                    SubtitleProcessData(
                        index=i + 1,
                        original_text=seg.text,
                        previous_context=previous,
                        next_context=following,
                    )
                )

            # 分批处理字幕
            chunks = self._split_chunks(translate_data_list)

            # 多线程翻译
            translated_list = self._parallel_translate(chunks)
            self.check_cancelled()

            # 设置Subtitle segment的翻译文本
            new_segments = self._set_segments_translated_text(
                asr_data.segments, translated_list
            )

            missing = [
                str(i)
                for i, seg in enumerate(new_segments, 1)
                if seg.text.strip() and not seg.translated_text.strip()
            ]
            if missing and not (allow_partial and any(seg.translated_text.strip() for seg in new_segments)):
                preview = ", ".join(missing[:10])
                suffix = "..." if len(missing) > 10 else ""
                raise RuntimeError(
                    f"翻译不完整：{len(missing)} 条字幕没有译文（序号 {preview}{suffix}），"
                    "已停止输出。请在字幕表核对对应时间和原文后重试。"
                )

            asr_data = ASRData(new_segments)
            return normalize_translated_script(asr_data, self.target_language)
        except CancelledError:
            raise
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            raise RuntimeError(f"Translation failed: {str(e)}")

    def _split_chunks(
        self, translate_data_list: List[SubtitleProcessData]
    ) -> List[List[SubtitleProcessData]]:
        """将字幕分割成块"""
        return [
            translate_data_list[i : i + self.batch_num]
            for i in range(0, len(translate_data_list), self.batch_num)
        ]

    def _parallel_translate(
        self, chunks: List[List[SubtitleProcessData]]
    ) -> List[SubtitleProcessData]:
        """并行翻译All块"""
        future_to_chunk = {}
        translated_list = []
        failed_indices = []
        first_error = None
        total_segments = sum(len(c) for c in chunks)

        for chunk in chunks:
            self.check_cancelled()
            future = self.executor.submit(self._safe_translate_chunk, chunk)
            future_to_chunk[future] = chunk

        for future in as_completed(future_to_chunk):
            self.check_cancelled()
            if not self.is_running:
                break
            try:
                result = future.result()
            except CancelledError:
                raise
            except Exception as e:
                if first_error is None:
                    first_error = e
                logger.error(f"Translation chunk failed: {e}")
                failed_indices.extend(cue.index for cue in future_to_chunk[future])
                continue
            translated_list.extend(result)
            if self.update_callback:
                self.update_callback(result)

        if self.allow_partial and first_error is not None and not translated_list:
            raise RuntimeError(f"沒有可用的譯文：{first_error}") from first_error
        # Preserve strict behavior for callers without an interactive review step.
        if failed_indices and total_segments > 0:
            failed_count = len(failed_indices)
            fail_rate = failed_count / total_segments
            if fail_rate >= 0.5 and not self.allow_partial:
                preview = ", ".join(map(str, sorted(failed_indices)[:10]))
                raise RuntimeError(
                    f"翻译不完整：{failed_count}/{total_segments} 条字幕翻译失败 "
                    f"({fail_rate:.0%}，序号 {preview})。已停止输出，请核对字幕表或翻译服务后重试。"
                )
            elif failed_count > 0:
                logger.warning(f"Translation partially failed: {failed_count}/{total_segments} segments")

        return translated_list

    def _get_cache_key(self, chunk: List[SubtitleProcessData]) -> str:
        """生成缓存键"""
        class_name = self.__class__.__name__
        chunk_key = generate_cache_key(chunk)
        lang = self.target_language.value
        return f"{class_name}:{chunk_key}:{lang}"

    def _safe_translate_chunk(
        self, chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """安全的翻译块"""
        try:
            self.check_cancelled()
            cache_key = self._get_cache_key(chunk)
            cached_result = None
            if not self.force_retry:
                try:
                    cached_result = self._cache.get(cache_key, default=None)
                except Exception:
                    self._cache.delete(cache_key)
            if cached_result is not None:
                cached_unfinished = [
                    data.index
                    for data in cached_result
                    if data.original_text.strip() and translation_needs_review(data.translated_text)
                ]
                if not cached_unfinished:
                    return cached_result
                logger.warning(
                    "Discarding incomplete translation cache for segments: "
                    f"{cached_unfinished}"
                )
                self._cache.delete(cache_key)

            last_error: Optional[Exception] = None
            partial_results: dict[int, SubtitleProcessData] = {}
            for attempt in range(1, 4):
                try:
                    self.check_cancelled()
                    result = self._translate_chunk(chunk)
                    self.check_cancelled()
                    missing = [
                        data.index
                        for data in result
                        if data.original_text.strip()
                        and not data.translated_text.strip()
                    ]
                    if missing:
                        if self.allow_partial:
                            partial_results.update(
                                (cue.index, replace(cue))
                                for cue in result
                                if cue.translated_text.strip()
                            )
                        raise RuntimeError(
                            f"Missing translated text for segments: {missing}"
                        )

                    if not any(
                        translation_needs_review(cue.translated_text) for cue in result
                    ):
                        self._cache.set(cache_key, result, expire=86400 * 7)
                    return result
                except CancelledError:
                    raise
                except Exception as e:
                    self.check_cancelled()
                    last_error = e
                    if attempt < 3:
                        logger.warning(
                            f"Translation attempt {attempt}/3 failed: {e}; retrying"
                        )
                        if self.cancel_event is not None:
                            self.cancel_event.wait(0.5 * attempt)
                        else:
                            time.sleep(0.5 * attempt)

            if partial_results:
                return list(partial_results.values())
            assert last_error is not None
            raise last_error

        except CancelledError:
            raise
        except Exception as e:
            logger.exception(f"Translation failed: {str(e)}")
            raise

    @staticmethod
    def _set_segments_translated_text(
        original_segments: List[ASRDataSeg], translated_list: List[SubtitleProcessData]
    ) -> List[ASRDataSeg]:
        """设置Subtitle segment的翻译文本"""
        # 创建索引到翻译文本的映射
        translation_map = {data.index: data.translated_text for data in translated_list}

        for i, seg in enumerate(original_segments, 1):
            if i not in translation_map:
                logger.error(f"Subtitle segment {i} has no translation")
            seg.translated_text = translation_map.get(i, "")

        return original_segments

    @abstractmethod
    def _translate_chunk(
        self, subtitle_chunk: List[SubtitleProcessData]
    ) -> List[SubtitleProcessData]:
        """翻译字幕块"""
        pass

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError()

    def stop(self, *, wait: bool = False):
        """停止翻译器"""
        if not self.is_running:
            return

        self.is_running = False
        atexit.unregister(self.stop)
        if hasattr(self, "executor") and self.executor is not None:
            try:
                self.executor.shutdown(wait=wait, cancel_futures=True)
            except Exception as e:
                logger.error(f"Error closing thread pool: {str(e)}")
            finally:
                self.executor = None
