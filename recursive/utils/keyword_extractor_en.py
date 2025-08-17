#coding: utf8
import re
import yake
from collections import OrderedDict, Counter
from markdown import markdown
from bs4 import BeautifulSoup
from typing import Optional, List, Dict
from loguru import logger

# 通用英文停用词
COMMON_ENGLISH_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "when", "where", "how",
    "who", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both",
    "through", "about", "for", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "ought", "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers",
    "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"
}

# 针对小说设计/正文的停用词 (信息量低的结构性或通用描述词)
STOP_WORDS_STORY = {
    "aspect", "issue", "situation", "content", "part", "way", "method", "need", "carry", "provide",
    "support", "achieve", "function", "module", "interface", "parameter", "return", "type", "data", "information",
    "user", "system", "design", "document", "background", "setting", "description", "goal", "scope", "introduction",
    "overview", "chapter", "scene", "main", "core", "relationship", "development", "first", "second", "third",
    "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"
}

# 针对报告的停用词
STOP_WORDS_REPORT = {
    "analysis", "summary", "recommendation", "conclusion", "abstract", "introduction", "methodology", "data_source",
    "chart", "appendix", "indicator", "trend", "evaluation", "strategy", "risk", "opportunity", "market", "customer",
    "competitor", "quarter", "annual", "year_on_year", "month_on_month", "growth", "decline"
}

# 针对工具书/教科书的停用词
STOP_WORDS_BOOK = {
    "definition", "theorem", "formula", "lemma", "proof", "corollary", "exercise", "problem", "question", "case",
    "example", "figure", "diagram", "introduction", "reference", "index", "table_of_contents", "preface", "acknowledgment",
    "concept", "principle", "assumption", "step", "point"
}

class KeywordExtractorEn:
    """
    一个集成了YAKE关键词提取、关键词优先级和增强停用词表的英文关键词提取器。
    通过'mode'参数支持不同写作场景（小说、报告、工具书）的优化。
    """
    def __init__(self, mode: str = 'story', stop_words=None, user_dict_paths: Optional[List[str]] = None, yake_params: Optional[Dict] = None):
        """
        初始化提取器。

        :param mode: 提取模式，可选值为 'story', 'report', 'book'。
                     根据不同模式，会自动加载相应的领域停用词。
        :param stop_words: 可选，自定义停用词集合。默认为 COMMON_ENGLISH_STOP_WORDS。
        :param user_dict_paths: 可选，自定义词典的路径列表。对于不同项目，强烈建议提供一个词典，包含所有核心的【专有名词】。

                                - 小说(story)模式示例:
                                  - 角色名: Harry Potter, Hermione Granger
                                  - 地点名: Hogwarts, Diagon Alley
                                  - 魔法/技能名: Expelliarmus, Patronus

                                - 报告(report)模式示例:
                                  - 公司/产品名: Google, Android, AI Assistant
                                  - 项目代号: Project Maven, AlphaGo
                                  - 行业术语: Machine Learning, NLP, API

                                - 工具书(book)模式示例:
                                  - 学科术语: Quantum Mechanics, Algorithms
                                  - 人物/事件: Albert Einstein, World War II
                                  - 定理/模型: Newton's Laws, Markov Chain
        """
        if stop_words:
            self.stop_words = stop_words
        else:
            mode_specific_stopwords = set()
            normalized_mode = mode.lower()
            if normalized_mode == 'story':
                mode_specific_stopwords = STOP_WORDS_STORY
            elif normalized_mode == 'report':
                mode_specific_stopwords = STOP_WORDS_REPORT
            elif normalized_mode == 'book':
                mode_specific_stopwords = STOP_WORDS_BOOK
            else:
                mode_specific_stopwords = STOP_WORDS_STORY
                logger.warning(f"Unsupported mode: {mode}, using 'story' mode instead")

            self.stop_words = COMMON_ENGLISH_STOP_WORDS.union(mode_specific_stopwords)
            logger.info(f"Initialized with {len(self.stop_words)} stop words for '{normalized_mode}' mode")

        # 用户词典目前对YAKE不适用，但我们可以记录加载尝试
        if user_dict_paths:
            for path in user_dict_paths:
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word and len(word) > 1:
                                self.user_dict.add(word.lower())
                    logger.info(f"Loaded {len(self.user_dict)} user-defined words from {path}")
                except FileNotFoundError:
                    logger.warning(f"User dictionary not found at {path}")
            logger.info(f"Total user-defined words: {len(self.user_dict)}")

        # Default YAKE parameters. Can be tuned.
        self.default_yake_params = {
            'lan': "en",
            'n': 3,
            'dedupLim': 0.85,
            'dedupFunc': 'seqm',
            'windowsSize': 2,
            'top': 20
        }
        if yake_params:
            self.default_yake_params.update(yake_params)
        logger.debug(f"YAKE parameters: {self.default_yake_params}")

    def _clean_markdown(self, markdown_text: str) -> str:
        """
        Converts markdown text to plain text.
        Removes HTML tags and other markdown syntax.
        """
        if not markdown_text:
            return ""
        # Convert markdown to HTML
        html = markdown(markdown_text)
        # Remove HTML tags
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

    def _process_text(self, text: str) -> list[str]:
        """
        处理文本，过滤掉停用词。
        """
        if not text or not isinstance(text, str):
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 简单分词 - 这里使用空格分词，因为YAKE会处理n-gram
        words = text.split()
        
        # 过滤停用词和长度小于2的词
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return filtered_words

    def extract_from_text(self, text: str, top_k: Optional[int] = None) -> list[str]:
        """
        从纯文本(如小说正文、报告段落)中提取关键词。
        此方法基于YAKE算法来确定关键词的重要性，特别适用于长篇、非结构化的文本内容。

        :param text: 待提取的纯文本内容。
        :param top_k: 可选，返回最重要的k个关键词。如果为None，则返回默认数量。
        :return: 按重要性降序排列的关键词列表。
        """
        if not text or not isinstance(text, str):
            return []
        
        # 过滤停用词
        processed_text = ' '.join(self._process_text(text))
        
        # 使用适当的参数创建YAKE提取器
        params = self.default_yake_params.copy()
        if top_k is not None:
            params['top'] = top_k
        
        kw_extractor = yake.KeywordExtractor(**params)
        keywords = kw_extractor.extract_keywords(processed_text)
        
        # 优先保留用户词典中的关键词
        final_keywords = []
        user_keywords_added = set()
        
        # 首先添加用户词典中的关键词
        if hasattr(self, 'user_dict') and self.user_dict:
            for kw, score in keywords:
                if kw.lower() in self.user_dict and kw not in user_keywords_added:
                    final_keywords.append(kw)
                    user_keywords_added.add(kw)
                    if top_k and len(final_keywords) >= top_k:
                        break
        
        # 然后添加其他关键词
        if top_k is None or len(final_keywords) < top_k:
            for kw, score in keywords:
                if kw not in user_keywords_added:
                    final_keywords.append(kw)
                    if top_k and len(final_keywords) >= top_k:
                        break
        
        return final_keywords

    def extract_from_markdown(self, markdown_text: str, top_k: Optional[int] = None) -> list[str]:
        """
        从Markdown格式的内容(如小说大纲、设计文档、报告结构)中提取关键词。

        此方法的核心优势是利用Markdown的结构化信息（如标题、加粗、列表）来区分关键词的优先级。
        特别适用于那些作者已经通过格式化来强调重点的文档。
        """
        if not markdown_text or not isinstance(markdown_text, str):
            return []

        high_priority_texts = []
        low_priority_texts = []

        # 1. 提取标题 (高优先级)
        headers = re.findall(r'^\s*#+\s*(.+)', markdown_text, re.MULTILINE)
        high_priority_texts.extend(headers)

        # 2. 提取加粗文本 (高优先级)
        bold_texts = re.findall(r'\*\*(.+?)\*\*', markdown_text)
        high_priority_texts.extend(bold_texts)

        # 3. 优化表格标题提取 (高优先级)
        # 查找后面跟着分隔符行的表格头
        table_header_pattern = re.compile(r'^\s*\|(.+)\|\s*\n\s*\|(?:\s*:?--+:?\s*\|)+', re.MULTILINE)
        for match in table_header_pattern.finditer(markdown_text):
            header_line = match.group(1)
            cleaned_headers = [h.strip() for h in header_line.split('|') if h.strip()]
            high_priority_texts.extend(cleaned_headers)

        # 4. 提取列表项 (低优先级)
        list_items = re.findall(r'^\s*[\*\-]\s+(.+)', markdown_text, re.MULTILINE)
        list_items.extend(re.findall(r'^\s*\d+\.\s+(.+)', markdown_text, re.MULTILINE))
        low_priority_texts.extend(list_items)

        # 5. 提取斜体文本 (低优先级)
        # 使用非贪婪匹配，并过滤掉可能是加粗文本的错误匹配
        italic_texts = re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', markdown_text)
        low_priority_texts.extend(italic_texts)

        # 6. 保留对Mermaid图表等特殊格式的支持 (低优先级)
        mermaid_blocks = re.findall(r'```mermaid\n(.+?)\n```', markdown_text, re.DOTALL)
        for block in mermaid_blocks:
            # 简单提取图表中的文本节点作为关键词
            mermaid_keywords = re.findall(r'(\w+)\s*\[', block)
            mermaid_keywords.extend(re.findall(r'"(.*?)"', block))
            low_priority_texts.extend(mermaid_keywords)

        # --- 处理和优先级处理 ---

        # 处理高优先级文本
        high_priority_full_text = " ".join(high_priority_texts)
        high_priority_keywords = self.extract_from_text(high_priority_full_text)

        # 处理低优先级文本
        low_priority_full_text = " ".join(low_priority_texts)
        low_priority_keywords = self.extract_from_text(low_priority_full_text)

        # 合并关键词，高优先级在前，并使用OrderedDict去重
        final_keywords = OrderedDict()
        for keyword in high_priority_keywords:
            final_keywords[keyword] = None
        
        for keyword in low_priority_keywords:
            # 只有当关键词不在高优先级列表中时才添加
            if keyword not in final_keywords:
                final_keywords[keyword] = None

        # 如果指定了top_k，返回前top_k个关键词
        result = list(final_keywords.keys())
        if top_k is not None:
            result = result[:top_k]

        return result