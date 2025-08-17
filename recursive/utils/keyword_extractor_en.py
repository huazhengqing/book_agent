#coding: utf8
import re
from collections import defaultdict
from typing import List
import stopwordsiso
from loguru import logger
from keybert import KeyBERT
from markdown import markdown
import diskcache as dc
import os
import hashlib
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize




# 英文：
# all-MiniLM-L6-v2（轻量高效，适合大多数场景）
# all-mpnet-base-v2（精度更高，但速度稍慢）
# 多语言：
# paraphrase-multilingual-MiniLM-L12-v2（轻量，支持 100 + 语言）
# xlm-r-bert-base-nli-stsb-mean-tokens（支持语言更多，精度较高）




class KeywordExtractorEn:
    def __init__(self):
        self.model = KeyBERT(model="all-MiniLM-L6-v2", nr_processes=os.cpu_count()//2)
        self.chunk_size = 1500  # 英文字符数 (约对应 450-500 tokens)
        self.chunk_overlap = 200  # 英文字符重叠数

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        self.base_stop_words = set(stopwordsiso.stopwords("en"))

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(current_dir, ".cache", "keyword_extractor_en")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = dc.Cache(cache_dir, size_limit=2 * 1024 * 1024 * 1024)


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
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                use_mmr=True,
                diversity=0.7,
                top_n=top_k,
                batch_size=32
            )

            for idx, keywords_with_scores in enumerate(batch_results):
                for kw, score in keywords_with_scores:
                    all_keywords_with_scores[kw] += score
                    keyword_chunk_count[kw] += 1

            sorted_keywords = sorted(all_keywords_with_scores.items(), key=lambda x: x[1], reverse=True)
            final_keywords = [kw for kw, _ in sorted_keywords][:top_k]

            self.cache.set(cache_key, final_keywords)
            logger.info(f"extract_from_text() final_keywords={final_keywords}")
            return final_keywords
        except ValueError as e:
            logger.error(f"Invalid value during keyword extraction: {e}")
            return []
        except RuntimeError as e:
            logger.error(f"Runtime error during model inference: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during keyword extraction: {e}")
            return []

    def extract_from_markdown(self, markdown_content: str, top_k: int = 30) -> list[str]:
        """
        从Markdown格式的内容(小说大纲、设计方案)中提取关键词
        """
        if not markdown_content or not markdown_content.strip():
            return []

        text_content = self.clean_markdown(markdown_content)
        return self.extract_from_text(text_content, top_k)

    def clean_markdown(self, markdown_text: str) -> str:
        html = markdown(markdown_text)
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # def clean_markdown(self, markdown_text: str) -> str:
    #     text = re.sub(r'#{1,6}\s*', '', markdown_text)  # 移除标题
    #     text = re.sub(r'\*\*?|__?', '', text)          # 移除粗/斜体
    #     text = re.sub(r'\[.*?\]\(.*?\)', '', text)    # 移除链接
    #     text = re.sub(r'\s+', ' ', text).strip()
    #     return text

    def split_long_text(self, text):
        # 基于NLTK的句子感知分块
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ''
        
        for sentence in sentences:
            # 计算包含当前句子的预估长度
            estimated_length = len(current_chunk) + len(sentence)
            
            if estimated_length > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    # 保留最后200词作为重叠上下文
                    overlap_words = current_chunk.split()[-self.chunk_overlap//4:]
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                else:  # 处理超长单句
                    chunk = sentence[:self.chunk_size]
                    chunks.append(chunk)
                    current_chunk = sentence[len(chunk)-self.chunk_overlap:] 
            else:
                current_chunk += ' ' + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        return chunks

    def preprocess_chunk(self, chunk):
        words = re.findall(r'\b[\w-]{2,}\b', chunk, flags=re.IGNORECASE)
        filtered = [w.lower() for w in words if w.lower() not in self.base_stop_words]
        return ' '.join(filtered)





keyword_extractor_en = KeywordExtractorEn()




