#coding: utf8
from copy import deepcopy
from collections import defaultdict
import re
from recursive.cache import Cache
import os
from loguru import logger
from recursive.llm.litellm_proxy import LiteLLMProxy
import json
from mem0 import Memory as Mem0Memory
from datetime import datetime
import time
import diskcache as dc
from recursive.agent.prompts.story_zh.mem import (
    MEM_STORY_FACT,
    MEM_STORY_UPDATE
)


class Mem0:
    def __init__(self, root_node, config):
        # 确定写作模式（story, book, report）
        self.writing_mode = config.get("writing_mode", "story")
        self.root_node = root_node
        
        # 使用diskcache作为查询缓存
        cache_dir = f"./.mem0/cache_{self.root_node.hashkey}"
        os.makedirs(cache_dir, exist_ok=True)
        self._query_cache = dc.Cache(cache_dir, size_limit=100*1024*1024)  # 100MB缓存限制
        

        self.mem0_config = {
            # "vector_store": {
            #     "provider": "chroma",
            #     "config": {
            #         "collection_name": f"story_{self.root_node.hashkey}",
            #         "path": "./.mem0/chroma_db"
            #     }
            # },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "mem0",
                    "host": "localhost",
                    "port": 6333,
                    "embedding_model_dims": int(os.getenv("embedding_dims"))
                }
            },
            # "vector_store": {
            #     "provider": "qdrant",
            #     "config": {
            #         "collection_name": "mem0",
            #         "host": "https://6549cdc3-556d-4740-9f26-ce8aeab31c84.europe-west3-0.gcp.cloud.qdrant.io",
            #         "port": 6333,
            #         "api_key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.z8CCJaOCx0_nIyKTLbJ3sDq-StXUmChccyQCmd2wbhE"
            #         "url": "https://6549cdc3-556d-4740-9f26-ce8aeab31c84.europe-west3-0.gcp.cloud.qdrant.io",
            #         "embedding_model_dims": int(os.getenv("embedding_dims")),
            #         # "client": "",
            #         # "path": "",
            #         # "on_disk": True
            #     }
            # },
            # "llm": {
            #     "provider": "openai",
            #     "config": {
            #         "model": "deepseek-ai/DeepSeek-V3",
            #         "temperature": 0.0,
            #         "max_tokens": 131072
            #     }
            # },
            "llm": {
                "provider": "litellm",
                "config": {
                    "model": os.getenv("fast_model"),
                    "temperature": 0.0,
                    "max_tokens": 131072,
                    "caching": True,
                    "max_completion_tokens": 131072,
                    "timeout": 300,
                    "num_retries": 2,
                    "respect_retry_after": True,
                    "fallbacks": [
                        "openrouter/deepseek/deepseek-chat-v3-0324:free",
                        "openai/deepseek-ai/DeepSeek-V3"
                        ]
                }
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "openai_base_url": os.getenv("embedder_BASE_URL"),
                    "api_key": os.getenv("embedder_API_KEY"),
                    "model": os.getenv("embedder_model"),
                    "embedding_dims": int(os.getenv("embedding_dims"))
                }
            },
            "graph_store": {
                "provider": "memgraph", 
                "config": {
                    "url": os.getenv("memgraph_url"),
                    "username": os.getenv("memgraph_username"),
                    "password": os.getenv("memgraph_password")
                }
            },
            # "history_db_path": "./.mem0/history.db",
            "custom_fact_extraction_prompt": MEM_STORY_FACT,
            "custom_update_memory_prompt": MEM_STORY_UPDATE
        }
        # if self.writing_mode != "story":
        #     self.config["custom_fact_extraction_prompt"] = MEM_STORY_FACT
        #     self.config["custom_update_memory_prompt"] = MEM_STORY_UPDATE

        # temp = os.getenv("OPENROUTER_API_KEY")
        # if self.mem0_config["llm"]["provider"] == "openai":
        #     # os.environ["OPENROUTER_API_KEY"] = ""
        #     if os.environ.get("OPENROUTER_API_KEY"):
        #         self.mem0_config["llm"]["config"].update({
        #             "model": "deepseek/deepseek-chat-v3-0324:free"
        #         })
        #     else:
        #         self.mem0_config["llm"]["config"].update({
        #             "model": "deepseek-ai/DeepSeek-V3"
        #         })

        
        self.client = Mem0Memory.from_config(config_dict=self.mem0_config)
        
        # os.environ["OPENROUTER_API_KEY"] = temp
        
        self.user_id_pre = f"{self.writing_mode}_{self.root_node.hashkey}"

        # self.llm_client = LiteLLMProxy()
        # self.fast_model = os.environ.get("fast_model")



    

    def add(self, content, content_type, task_info):
        if self.writing_mode == "story":
            return self.story_add(content, content_type, task_info)

    def get_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        if self.writing_mode == "story":
            return self.get_story_outer_graph_dependent(task_info, same_graph_dependent_designs, latest_content)

    def get_content(self, task_info, same_graph_dependent_designs, latest_content):
        if self.writing_mode == "story":
            return self.get_story_content(task_info, same_graph_dependent_designs, latest_content)

    def get_full_plan(self, task_info):
        if self.writing_mode == "story":
            return self.get_story_full_plan(task_info)




    def story_add(self, content, content_type, task_info):
        task_id = task_info.get("id")
        task_type = task_info.get("task_type")
        task_goal = task_info.get("goal")
        dependency = task_info.get("dependency")
        task_str = json.dumps(task_info, ensure_ascii=False)

        if content_type == "story_content":
            category = "story"
        elif content_type in ["task_update", "task_decomposition", "design_result"]:
            category = "design"

        mem0_content = ""
        if content_type == "story_content":
            mem0_content = content
        elif content_type == "task_decomposition":
            mem0_content = f"""任务：\n{task_str}\n规划分解结果：\n{content}"""
        elif content_type == "design_result":
            mem0_content = f"""任务：\n{task_str}\n设计结果：\n{content}"""
        elif content_type == "task_update":
            mem0_content = f"""任务更新：\n{content}"""

        parent_task_id = ".".join(task_id.split(".")[:-1]) if task_id and "." in task_id else ""
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        mem_metadata = {
            "category": category,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "hierarchy_level": len(task_id.split(".")),
            "parent_task_id": parent_task_id,
            "dependency": dependency_str,
            "content_length": len(mem0_content),
            "content_hash": hash(mem0_content) % 10000
        }
        
        logger.info(f"mem0 story_add() mem0_content=\n{mem0_content}\n mem_metadata=\n{mem_metadata}")
        self.client.add(
            mem0_content,
            user_id=f"{self.user_id_pre}_{category}",
            metadata=mem_metadata
        )





    def search(self, querys, category, limit=1, filters=None):
        query = " ".join(querys)
        
        cache_key = f"{self.user_id_pre}_{category}_{query}_{limit}_{str(filters)}"
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        logger.info(f"mem0 search() {category} query=\n{query}")
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_{category}",
            limit=limit,
            filters=filters
        )
        logger.info(f"mem0 search() results=\n{results}")
        self._query_cache[cache_key] = results
        return results
        





    def _extract_keywords_from_markdown(self, markdown_content):
        """
        从 Markdown 格式的设计结果中提取关键词（优化中文处理）
        Args:
            markdown_content: Markdown 格式的文本内容
        Returns:
            list: 提取的关键词列表
        """
        if not markdown_content or not isinstance(markdown_content, str):
            return []
        
        keywords = []
        
        # 1. 提取标题关键词 (### 标题)
        # 支持中文标题
        headings = re.findall(r'#+\s+([^\n]+)', markdown_content)
        keywords.extend(headings)
        
        # 2. 提取加粗文本 (**加粗**)
        bold_text = re.findall(r'\*\*([^\*]+)\*\*', markdown_content)
        keywords.extend(bold_text)
        
        # 3. 提取斜体文本 (*斜体*)
        italic_text = re.findall(r'\*([^*]+)\*', markdown_content)
        keywords.extend(italic_text)
        
        # 4. 提取列表项 (- 或 * 开头)
        list_items = re.findall(r'[-\*]\s+([^\n]+)', markdown_content)
        keywords.extend(list_items)
        
        # 5. 提取表格标题
        table_headers = re.findall(r'\|\s*([^|]+)\s*\|', markdown_content)
        keywords.extend(table_headers)
        
        # 6. 提取代码块中的关键词 (保留但优化)
        code_blocks = re.findall(r'```[\s\S]*?```', markdown_content)
        for block in code_blocks:
            # 移除代码块标记和语言标识
            clean_block = re.sub(r'```[a-z]*\n|```', '', block)
            # 提取可能的关键词（变量名、函数名等）
            # 针对中文代码注释进行优化
            code_keywords = re.findall(r'[a-zA-Z_]+|[\u4e00-\u9fa5]+', clean_block)
            keywords.extend(code_keywords)
        
        # 7. 提取 Mermaid 图表中的关键词
        mermaid_blocks = re.findall(r'```mermaid[\s\S]*?```', markdown_content)
        for block in mermaid_blocks:
            # 提取节点和关系，支持中文节点名
            nodes = re.findall(r'([\u4e00-\u9fa5\w]+)\s*\[', block)
            keywords.extend(nodes)
        
        # 8. 清理和过滤关键词
        filtered_keywords = []
        # 中文停用词表
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '那个', '时候', '觉得', '开始', '这样', '起来', '没有', '用', '过', '又',
            '现在', '天气', '今天', '明天', '昨天', '这里', '那里', '一些', '一点', '一下', '下去', '出来', '回去', '东西', '事情',
            '他们', '她们', '我们', '你们', '它们', '大家', '人们', '这里', '那里', '上面', '下面', '里面', '外面', '地方', '时间',
            '的话', '所以', '因为', '但是', '不过', '然后', '接着', '终于', '刚刚', '已经', '正在', '将要', '能够', '可以', '应该',
            '必须', '可能', '也许', '大概', '非常', '特别', '十分', '很', '更', '最', '太', '还', '再', '也', '又', '只', '才', '就',
            '都', '全', '总', '共', '一共', '一起', '一同', '一道', '一样', '同样', '另外', '其他', '别的', '其余', '所有', '全部',
            '任何', '每', '各', '每个', '各个', '各种', '各样', '怎样', '怎么', '如何', '多少', '几', '哪个', '哪些', '什么', '为什么'
        }
        
        for keyword in keywords:
            # 去除空白字符
            keyword = keyword.strip()
            # 过滤太短的关键词和停用词
            # 中文词语长度至少为1，英文单词长度至少为2
            if (len(keyword) > 1 and keyword.lower() not in stop_words) or (len(keyword) == 1 and re.match(r'[\u4e00-\u9fa5]', keyword)):
                filtered_keywords.append(keyword)
        
        # 去重但保持顺序
        return list(dict.fromkeys(filtered_keywords))
        


    def _extract_keywords_from_text(self, text_content):
        """
        从纯文本中提取关键词
        """
        if not text_content or not isinstance(text_content, str):
            return []
        
        # 使用正则表达式提取中文词语和英文单词
        keywords = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z]+', text_content)
        
        # 过滤停用词和短词
        stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
            '会', '着', '没有', '看', '好', '自己', '这', '那个', '时候', '觉得', '开始', '这样', '起来', '没有', '用', '过', '又',
            '现在', '天气', '今天', '明天', '昨天', '这里', '那里', '一些', '一点', '一下', '下去', '出来', '回去', '东西', '事情',
            '他们', '她们', '我们', '你们', '它们', '大家', '人们', '这里', '那里', '上面', '下面', '里面', '外面', '地方', '时间'
        }
        
        filtered_keywords = []
        for keyword in keywords:
            # 中文单字非停用词可保留，英文单词至少2个字符
            if (len(keyword) == 1 and re.match(r'[\u4e00-\u9fa5]', keyword) and keyword not in stop_words) or (len(keyword) > 1 and keyword not in stop_words):
                filtered_keywords.append(keyword)
        
        # 去重并返回
        return list(dict.fromkeys(filtered_keywords))




    def get_story_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        """检索上层设计结果，替换to_run_outer_graph_dependent"""
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        
        all_queries = []

        all_queries.append(task_goal)

        overview_keywords = [
            "故事进展", "角色状态", "情节发展", "主要角色", "关键事件", 
            "时间线", "角色关系", "冲突", "转折", "背景设定"
        ]
        all_queries.extend(overview_keywords)
        
        # 提取层级信息（卷、幕、章、场景、节拍）
        level_patterns = [
            (r'第(\d+)卷', '卷'),
            (r'第(\d+)幕', '幕'), 
            (r'第(\d+)章', '章'),
            (r'场景(\d+)', '场景'),
            (r'节拍(\d+)', '节拍')
        ]
        for pattern, level_name in level_patterns:
            matches = re.findall(pattern, task_goal)
            for num in matches:
                # 精确层级匹配
                all_queries.append(f"第{num}{level_name}")
                # 前后层级上下文（用于获取连续性）
                try:
                    num_int = int(num)
                    if num_int > 1:
                        all_queries.append(f"第{num_int-1}{level_name}")
                    all_queries.append(f"第{num_int+1}{level_name}")
                except:
                    pass
        
        # 提取任务ID相关上下文
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id}")
        
        if latest_content and isinstance(latest_content, str):
            all_queries.extend(self._extract_keywords_from_text(latest_content))

        all_queries.extend(self._extract_keywords_from_markdown(same_graph_dependent_designs))
        
        # 合并所有关键词并去重
        final_queries = list(dict.fromkeys(all_queries))

        # 执行检索
        all_results = self.search(final_queries, "design", limit=500)
        
        # 处理结果
        combined_results = []
        for result in all_results:
            memory_content = result.get('memory', '')
            if memory_content:
                combined_results.append(memory_content)
        return "\n\n".join(combined_results)












    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        """
        检索已写的正文内容，替换memory.article
        目标：提供故事概览，让LLM了解故事进展、角色状态、情节发展
        """
        task_goal = task_info.get('goal', '')
        task_id = task_info.get('id', '')
        
        all_queries = []

        all_queries.append(task_goal)

        overview_keywords = [
            "故事进展", "角色状态", "情节发展", "主要角色", "关键事件", 
            "时间线", "角色关系", "冲突", "转折", "背景设定"
        ]
        all_queries.extend(overview_keywords)
        
        # 提取层级信息（卷、幕、章、场景、节拍）
        level_patterns = [
            (r'第(\d+)卷', '卷'),
            (r'第(\d+)幕', '幕'), 
            (r'第(\d+)章', '章'),
            (r'场景(\d+)', '场景'),
            (r'节拍(\d+)', '节拍')
        ]
        for pattern, level_name in level_patterns:
            matches = re.findall(pattern, task_goal)
            for num in matches:
                # 精确层级匹配
                all_queries.append(f"第{num}{level_name}")
                # 前后层级上下文（用于获取连续性）
                try:
                    num_int = int(num)
                    if num_int > 1:
                        all_queries.append(f"第{num_int-1}{level_name}")
                    all_queries.append(f"第{num_int+1}{level_name}")
                except:
                    pass
        
        # 提取任务ID相关上下文
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id}")
        
        if latest_content and isinstance(latest_content, str):
            all_queries.extend(self._extract_keywords_from_text(latest_content))

        all_queries.extend(self._extract_keywords_from_markdown(same_graph_dependent_designs))
        
        # 合并所有关键词并去重
        final_queries = list(dict.fromkeys(all_queries))

        # 执行检索
        all_results = self.search(final_queries, "story", limit=500)
        
        # 处理结果
        combined_results = []
        for result in all_results:
            memory_content = result.get('memory', '')
            if memory_content:
                combined_results.append(memory_content)
        return "\n\n".join(combined_results)











    def get_story_full_plan(self, task_info):
        task_id = task_info.get("id", "")
        if not task_id:
            return ""
        
        to_task_ids = []
        current_id = task_id
        to_task_ids.append(current_id)
        while "." in current_id:
            current_id = ".".join(current_id.split(".")[:-1])
            to_task_ids.append(current_id)
        to_task_ids = sorted(to_task_ids, key=lambda x: len(x.split(".")))

        task_goals = []
        for pid in to_task_ids:
            results = self.client.search(
                query=f"任务id为{pid}的详细目标(goal)",
                user_id=f"{self.user_id_pre}_design",
                limit=1
            )
            results = results if isinstance(results, list) else results.get('results', [])
            if not results:
                continue
            task_goals.append(f"[{pid}]: {results[0].get('memory')}")
        
        return "\n".join(task_goals)


    








