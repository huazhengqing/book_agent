#coding: utf8
import re
import jieba
import jieba.analyse
from collections import OrderedDict, Counter
from typing import Optional, List, Dict
from loguru import logger

# 通用中文停用词
COMMON_CHINESE_STOP_WORDS = {
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们", "这", "那", "一", "个", "也", "有", "和", "人",
    "就", "不", "而", "所", "为", "之", "与", "以", "可", "到", "说", "中", "上", "下", "前", "后", "里", "外",
    "等", "于", "对", "能", "都", "地", "得", "着", "会", "自己", "我们", "你们", "他们", "她们", "它们",
    "这个", "那个", "一些", "什么", "怎么", "哪里", "哪个", "为什么", "如何", "如果", "但是", "所以", "因为",
    "或者", "并且", "而且", "然后", "于是", "关于", "对于", "以及", "此外", "另外", "其他", "相关", "例如", "包括",
    # 标点符号
    "、", "。", "，", "“", "”", "‘", "’", "；", "：", "？", "！", "（", "）", "【", "】", "《", "》", "—", "～",
    " ", "\n", "\t",
}

# 针对小说设计/正文的停用词 (信息量低的结构性或通用描述词)
STOP_WORDS_STORY = {
    "方面", "问题", "情况", "内容", "部分", "方式", "方法", "需要", "进行", "提供", "支持", "实现", "功能",
    "模块", "接口", "参数", "返回", "类型", "数据", "信息", "用户", "系统", "设计", "文档",
    "背景", "设定", "描述", "目标", "范围", "简介", "介绍", "概述", "章节", "场景", "主要", "核心", "关系", "发展",
    "第一", "第二", "第三", "第四", "第五", "第六", "第七", "第八", "第九", "第十"
}

# 针对报告的停用词
STOP_WORDS_REPORT = {
    "分析", "总结", "建议", "结论", "摘要", "引言", "方法论", "数据来源", "图表", "附录", "指标", "趋势", "评估",
    "策略", "风险", "机遇", "市场", "客户", "竞品", "季度", "年度", "同比", "环比", "增长", "下降"
}

# 针对工具书/教科书的停用词
STOP_WORDS_BOOK = {
    "定义", "定理", "公式", "引理", "证明", "推论", "练习", "习题", "思考题", "案例", "示例", "如图所示", "引言",
    "参考文献", "索引", "目录", "前言", "致谢", "概念", "原理", "假设", "步骤", "要点"
}

class KeywordExtractorZh:
    """
    一个集成了jieba分词、关键词优先级和增强停用词表的关键词提取器。
    通过'mode'参数支持不同写作场景（小说、报告、工具书）的优化。
    """
    def __init__(self, mode: str = 'story', stop_words=None, user_dict_paths: Optional[List[str]] = None, jieba_params: Optional[Dict] = None):
        """
        初始化提取器。

        :param mode: 提取模式，可选值为 'story', 'report', 'book'。
                     根据不同模式，会自动加载相应的领域停用词。
        :param stop_words: 可选，自定义停用词集合。默认为 CHINESE_STOP_WORDS。
        :param user_dict_paths: 可选，jieba自定义词典的路径列表。对于不同项目，强烈建议提供一个词典，包含所有核心的【专有名词】。
                                
                                - 小说(story)模式示例:
                                  - 角色名: 萧炎, 云韵
                                  - 地点名: 乌坦城, 魔兽山脉
                                  - 功法/技能名: 焚诀, 佛怒火莲
                                
                                - 报告(report)模式示例:
                                  - 公司/产品名: 腾讯, 微信, AI Agent平台
                                  - 项目代号: “灯塔”项目, “方舟”计划
                                  - 行业术语: AIGC, MLOps, DAU (日活跃用户)
                                
                                - 工具书(book)模式示例:
                                  - 学科术语: 梯度下降, 熵, 相对论
                                  - 人物/事件: 牛顿, 第一次工业革命
                                  - 定理/模型: 勾股定理, Transformer模型
                                
                                这能极大提高分词准确性，避免专有名词被错误切分。
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

            self.stop_words = COMMON_CHINESE_STOP_WORDS.union(mode_specific_stopwords)
            logger.info(f"Initialized with {len(self.stop_words)} stop words for '{normalized_mode}' mode")

        # 初始化用户词典集合
        self.user_dict = set()

        if user_dict_paths:
            for path in user_dict_paths:
                try:
                    jieba.load_userdict(path)
                    logger.info(f"Successfully loaded custom jieba dictionary from {path}")
                    # 同时加载到用户词典集合中
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                word = parts[0]
                                if word and len(word) > 1:
                                    self.user_dict.add(word)
                except FileNotFoundError:
                    logger.warning(f"Custom jieba dictionary not found at {path}")
            logger.info(f"Total user-defined words: {len(self.user_dict)}")

        # 设置jieba参数
        self.jieba_params = {
            'use_smart': True
        }
        if jieba_params:
            self.jieba_params.update(jieba_params)
        logger.debug(f"Jieba parameters: {self.jieba_params}")

    def _process_text_with_jieba(self, text: str) -> list[str]:
        """
        使用jieba对文本进行分词，并过滤掉停用词和单字。
        """
        if not text or not isinstance(text, str):
            return []
        
        # 设置jieba智能模式
        jieba.smart = self.jieba_params.get('use_smart', True)
        
        # 使用精确模式进行分词
        words = jieba.cut(text, cut_all=False)
        
        # 过滤停用词和长度小于2的词
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        
        return filtered_words

    def extract_from_text(self, text: str, top_k: Optional[int] = None) -> list[str]:
        """
        从纯文本(如小说正文、报告段落)中提取关键词。
        自动选择最合适的算法(TextRank)进行提取。

        :param text: 待提取的纯文本内容。
        :param top_k: 可选，返回最重要的k个关键词。如果为None，则返回默认数量(20)。
        :return: 按重要性降序排列的关键词列表。
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input for keyword extraction")
            return []
        
        # 确保top_k有合理值
        if top_k is None:
            top_k = 20
        
        # 处理文本
        processed_text = self._process_text_with_jieba(text)
        full_text = ' '.join(processed_text)
        
        # 使用TextRank算法作为默认算法
        keywords = jieba.analyse.textrank(full_text, topK=top_k, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
        logger.debug(f"Extracted top {len(keywords)} keywords using TextRank")
        
        # 优先保留用户词典中的关键词
        final_keywords = []
        user_keywords_added = set()
        
        # 首先添加用户词典中的关键词
        if self.user_dict:
            for kw in keywords:
                if kw in self.user_dict and kw not in user_keywords_added:
                    final_keywords.append(kw)
                    user_keywords_added.add(kw)
                    if len(final_keywords) >= top_k:
                        break
        
        # 然后添加其他关键词
        if len(final_keywords) < top_k:
            for kw in keywords:
                if kw not in user_keywords_added:
                    final_keywords.append(kw)
                    if len(final_keywords) >= top_k:
                        break
        
        return final_keywords

    def extract_from_markdown(self, markdown_content: str, top_k: Optional[int] = None) -> list[str]:
        """
        从Markdown格式的内容(如小说大纲、设计文档、报告结构)中提取关键词。

        此方法的核心优势是利用Markdown的结构化信息（如标题、加粗、列表）来区分关键词的优先级。
        它特别适用于那些作者已经通过格式化来强调重点的文档。
        这与基于纯词频的 `extract_from_text` 方法形成了互补。

        :param markdown_content: 待提取的Markdown内容。
        :param top_k: 可选，返回最重要的k个关键词。如果为None，则返回所有关键词。
        :return: 按优先级降序排列的关键词列表。
        """
        if not markdown_content or not isinstance(markdown_content, str):
            return []

        high_priority_texts = []
        low_priority_texts = []

        # 1. 提取标题 (高优先级)
        headers = re.findall(r'^\s*#+\s*(.+)', markdown_content, re.MULTILINE)
        high_priority_texts.extend(headers)

        # 2. 提取加粗文本 (高优先级)
        bold_texts = re.findall(r'\*\*(.+?)\*\*', markdown_content)
        high_priority_texts.extend(bold_texts)

        # 3. 优化表格标题提取 (高优先级)
        # 查找后面跟着分隔符行的表格头
        table_header_pattern = re.compile(r'^\s*\|(.+)\|\s*\n\s*\|(?:\s*:?--+:?\s*\|)+', re.MULTILINE)
        for match in table_header_pattern.finditer(markdown_content):
            header_line = match.group(1)
            cleaned_headers = [h.strip() for h in header_line.split('|') if h.strip()]
            high_priority_texts.extend(cleaned_headers)

        # 4. 提取列表项 (低优先级)
        list_items = re.findall(r'^\s*[\*\-]\s+(.+)', markdown_content, re.MULTILINE)
        list_items.extend(re.findall(r'^\s*\d+\.\s+(.+)', markdown_content, re.MULTILINE))
        low_priority_texts.extend(list_items)

        # 5. 提取斜体文本 (低优先级)
        # 使用非贪婪匹配，并过滤掉可能是加粗文本的错误匹配
        italic_texts = re.findall(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', markdown_content)
        low_priority_texts.extend(italic_texts)

        # 6. 保留对Mermaid图表等特殊格式的支持 (低优先级)
        mermaid_blocks = re.findall(r'```mermaid\n(.+?)\n```', markdown_content, re.DOTALL)
        for block in mermaid_blocks:
            # 简单提取图表中的文本节点作为关键词
            mermaid_keywords = re.findall(r'(\w+)\s*\[', block)
            mermaid_keywords.extend(re.findall(r'"(.*?)"', block))
            low_priority_texts.extend(mermaid_keywords)

        # --- 分词和优先级处理 ---

        # 处理高优先级文本
        high_priority_full_text = " ".join(high_priority_texts)
        high_priority_keywords = self._process_text_with_jieba(high_priority_full_text)

        # 处理低优先级文本
        low_priority_full_text = " ".join(low_priority_texts)
        low_priority_keywords = self._process_text_with_jieba(low_priority_full_text)

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
            logger.debug(f"Extracted top {top_k} keywords from markdown")
        else:
            logger.debug(f"Extracted all {len(result)} keywords from markdown")

        return result