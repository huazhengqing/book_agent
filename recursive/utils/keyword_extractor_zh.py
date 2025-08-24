#coding: utf8
import re
import math
from collections import defaultdict
from typing import List
import stopwordsiso
from loguru import logger
from keybert import KeyBERT
from markdown import markdown
import diskcache as dc
import os
import jieba
import hashlib
# import jieba.posseg as pseg
import multiprocessing
import threading


"""
# KeywordExtractorZh
- 中文关键词提取器，基于 KeyBERT 和 jieba 库
- 支持从文本和 Markdown 中提取关键词
- 实现文本分块处理、预处理和缓存功能
- 使用 BAAI/bge-small-zh 模型进行中文语义理解

# model
中文：
shibing624/text2vec-base-chinese（专为中文优化的通用模型）
BAAI/bge-small-zh（中文语义理解能力强，适合长文本）
多语言：
paraphrase-multilingual-MiniLM-L12-v2（轻量，支持 100 + 语言）
xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多，精度较高）
"""


###############################################################################


class KeywordExtractorZh:
    def __init__(self):
        self.model = None
        self.chunk_size = 700  # 中文字符数 (约对应 450-500 tokens)
        self.chunk_overlap = 100  # 中文字符重叠数
        self._model_lock = threading.Lock()

        threads = min(multiprocessing.cpu_count(), 4)
        jieba.enable_parallel(threads)

        self.base_stop_words = set(stopwordsiso.stopwords("zh"))

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        cache_dir = os.path.join(project_root, ".cache", "keyword_extractor_zh")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = dc.Cache(cache_dir, size_limit=1024 * 1024 * 300)

    def __del__(self):
        try:
            jieba.disable_parallel()
        except:
            pass

    def _ensure_model_initialized(self):
        if not self.model:
            with self._model_lock:
                if not self.model:
                    # self.model = KeyBERT(model="BAAI/bge-small-zh", nr_processes=os.cpu_count()//2)
                    self.model = KeyBERT(model="./models/bge-small-zh", nr_processes=os.cpu_count()//2)

    def extract_from_text(self, text: str, top_k: int = 30) -> List[str]:
        """
        从小说正文中提取关键词
        包含层级结构（全书、卷、幕、章、场景、节拍、段落）
        KeyBERT 批处理：
            - 输入：`docs` 参数传文本列表 `[text1, text2, ...]`
            - 返回：嵌套列表，每个子列表对应输入文本的关键词 `[(kw, score), ...]`
        """
        if not text or not text.strip():
            return []

        self._ensure_model_initialized()

        text_hash = hashlib.blake2b(text.encode('utf-8'), digest_size=32).hexdigest()
        cache_key = text_hash
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        try:
            chunks = self.split_long_text(text)
            all_keywords_with_scores = defaultdict(float)
            keyword_chunk_count = defaultdict(int)

            processed_chunks = []
            for chunk in chunks:
                processed_text = self.preprocess_chunk(chunk)
                if processed_text:
                    processed_chunks.append(processed_text)

            if not processed_chunks:
                return []

            batch_results = self.model.extract_keywords(
                processed_chunks,
                keyphrase_ngram_range=(1, 2),
                stop_words="chinese",
                use_mmr=True,
                diversity=0.6,
                top_n=top_k,
                batch_size=32
            )

            for idx, keywords_with_scores in enumerate(batch_results):
                for kw, score in keywords_with_scores:
                    all_keywords_with_scores[kw] += score
                    keyword_chunk_count[kw] += 1

            # 优化关键词权重计算算法
            chunk_count = len(processed_chunks)
            weighted_keywords = {}
            for kw, score in all_keywords_with_scores.items():
                freq_factor = math.sqrt(keyword_chunk_count[kw] / chunk_count)
                weighted_keywords[kw] = score * freq_factor

            # 按加权分数排序
            sorted_keywords = sorted(weighted_keywords.items(), key=lambda x: x[1], reverse=True)
            final_keywords = [kw for kw, _ in sorted_keywords][:top_k]

            self.cache.set(cache_key, final_keywords)
            logger.info(f"extract_from_text() final_keywords={final_keywords}")
            return final_keywords
        except Exception as e:
            logger.error(f"Unexpected error during keyword extraction: {e}")
            return []

    def extract_from_markdown(self, markdown_content: str, top_k: int = 30) -> list[str]:
        """
        从Markdown格式的内容(小说大纲、设计方案)中提取关键词
        """
        if not markdown_content or not markdown_content.strip():
            return []

        self._ensure_model_initialized()
        
        text_content = self.clean_markdown(markdown_content)
        return self.extract_from_text(text_content, top_k)

    def clean_markdown(self, markdown_text: str) -> str:
        html = markdown(markdown_text)
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def split_long_text(self, text):
        # 基于中文标点的句子感知分块
        sentences = re.split(r'([。！？；;!?])', text)
        chunks = []
        current_chunk = ''
        
        # 合并句子时保留分隔符
        for i in range(0, len(sentences)-1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
            
            # 动态调整块大小
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence[-self.chunk_overlap:]  # 保留重叠部分
                else:  # 处理超长单句
                    chunks.append(sentence[:self.chunk_size])
                    current_chunk = sentence[self.chunk_size - self.chunk_overlap:]
            else:
                current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        return [chunk for chunk in chunks if chunk]


    def preprocess_chunk(self, chunk):
        chunk = re.sub(r'[\u3000-\u303F\uff00-\uffef\u2018-\u201f]', ' ', chunk)
        words = jieba.lcut(chunk, cut_all=False)
        filtered = [w for w in words if w not in self.base_stop_words and len(w) > 1]
        return " ".join(filtered)


###############################################################################


keyword_extractor_zh = KeywordExtractorZh()




