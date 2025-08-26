#coding: utf8
import os
from loguru import logger
from recursive.llm.litellm_proxy import llm_client
import json
from mem0 import Memory as Mem0Memory
from datetime import datetime
from recursive.agent.prompts.story_zh.mem import (
    mem_story_fact_zh,
    mem_story_update_zh,
    mem_story_design_queries_zh_system,
    mem_story_design_queries_zh_user,
    mem_story_text_queries_zh_system,
    mem_story_text_queries_zh_user
)
from recursive.agent.prompts.report_zh.mem import (
    mem_report_fact_zh,
    mem_report_update_zh,
    mem_report_design_queries_zh_system,
    mem_report_design_queries_zh_user,
    mem_report_text_queries_zh_system,
    mem_report_text_queries_zh_user
)
import diskcache
import hashlib
from datetime import datetime


"""
# 引入 mem0 的终极目的
- 解决全量传输内容导致的成本过高和token限制问题
- 仅向 LLM 提供与当前任务最相关的信息，降低token消耗，同时保持写作上下文的连贯性
# mem0_wrapper.py
- 是 Mem0 内存系统的封装类，提供记忆添加、搜索和获取功能
- 集成 Qdrant 向量数据库和 Memgraph 图数据库
- mem0的文档 docs/llms-mem0.txt
- 分类存储：按 正文内容 和 设计方案 分别存储
- 自定义提示词：针对小说创作定制了事实提取和更新提示词，在 agent/prompts/story_zh 中的 mem.py 中的 mem_story_fact_zh  mem_story_update_zh
- 支持 story 、 book 、 report 三种写作模式和多语言支持
- 根据语言选择对应的关键词提取器
- 实现动态查询生成，用于检索设计库和正文库
## 工作流程
- 1. 记忆存储
    - mem0_wrapper.py 的 add 方法将内容（小说正文，分解结果，设计结果，任务更新）与元数据存入 Mem0
    - 使用自定义提示词：agent/prompts/story_zh 中的 mem.py 中的 mem_story_fact_zh、mem_story_update_zh 进行事实提取和更新
    - 故事正文是由 agent/prompts/story_zh 中的 writer.py 生成的
    - 设计结果是由 agent/prompts/story_zh 中的 reasoner.py 生成的
    - 任务分解更新是由 agent/prompts/story_zh 中的 planning.py 生成的
- 2. 查询生成
    - 通过 _generate_design_queries 和 _generate_text_queries 方法生成动态查询
    - 根据语言选择 keyword_extractor_zh 或 keyword_extractor_en 从最新内容和相关设计中提取关键词
- 3. 内容检索
    - search 方法使用生成的查询词从向量数据库中检索相关内容
    - get_story_outer_graph_dependent  是检索设计结果，它的目标是替换  agent/agents/regular.py  中的 get_llm_output 中的  to_run_outer_graph_dependent
    - get_story_content  是检索小说已经写的正文内容，它的目标是替换  agent/agents/regular.py 中的 get_llm_output 中的 memory.article
- 4. 结果应用
    - 检索结果作为上下文提供给 LLM 用于生成新的内容或设计
    - 检索结果在 agent/agents/regular.py 中的 get_llm_output 中的 prompt_args 中组装为上下文，传入 agent/prompts/story_zh 中的 planning.py、reasoner.py、writer.py
"""


###############################################################################


"""
分析、评估、审查 项目 的 RAG 的质量和效果，项目的目标是创作出爆款的超长篇网络小说，从项目角度，给出全面的报告，并指出其最大的优势和可以进一步强化的方向。


"""

class Mem0:
    def __init__(self, writing_mode, language):
        # 确定写作模式（story, book, report）
        self.writing_mode = writing_mode
        self.language = language
        self.mem0_config = {
            # "vector_store": {
            #     "provider": "chroma",
            #     "config": {
            #         "collection_name": f"story_{self.language}",
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
                    "model": 'openai/deepseek-ai/DeepSeek-V3',
                    "temperature": 0.0,
                    "max_tokens": 131072,
                    "caching": True,
                    "max_completion_tokens": 131072,
                    "timeout": 300,
                    "num_retries": 2,
                    "respect_retry_after": True,
                    "fallbacks": [
                        # 'openai/deepseek-ai/DeepSeek-V3',
                        'openrouter/deepseek/deepseek-chat-v3-0324:free',
                        # 'openrouter/qwen/qwen3-32b'
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
            # "history_db_path": "./.mem0/history.db"
        }
        prompt_config = {
            "story": (mem_story_fact_zh, mem_story_update_zh),
            "book": ("", ""), 
            "report": (mem_report_fact_zh, mem_report_update_zh) 
        }
        
        if self.writing_mode not in prompt_config:
            raise ValueError(f"不支持的写作模式: {self.writing_mode}")
        
        fact_prompt, update_prompt = prompt_config[self.writing_mode]
        if fact_prompt:
            self.mem0_config["custom_fact_extraction_prompt"] = fact_prompt
            self.mem0_config["custom_update_memory_prompt"] = update_prompt

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

        self._init_caches()
        
    def _init_caches(self):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        cache_configs = {
            'cache_text': ('mem0_text', 300),
            'cache_design': ('mem0_design', 300), 
            'cache_full_plan': ('mem0_full_plan', 100)
        }
        for cache_name, (subdir, size_mb) in cache_configs.items():
            cache_dir = os.path.join(project_root, ".cache", subdir)
            os.makedirs(cache_dir, exist_ok=True)
            setattr(self, cache_name, diskcache.Cache(cache_dir, size_limit=1024 * 1024 * size_mb))

    def add(self, hashkey, content, content_type, task_info):
        task_id = task_info.get("id")
        if not task_id or task_id == "" or task_id == "0":
            raise ValueError(f"任务信息中未找到任务ID {task_id} \n 任务信息: {task_info}")
        
        dependency = task_info.get("dependency", [])
        task_str = json.dumps(task_info, ensure_ascii=False)
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        logger.info(f"mem0 添加记忆 {task_info}")

        # 内容类型配置：{content_type: (category, format_template, cache_to_clear)}
        content_configs = {
            "text_content": ("text", lambda c, ts: c, "cache_text"),
            "task_decomposition": ("design", lambda c, ts: f"任务：\n{ts}\n规划分解结果：\n{c}", "cache_full_plan"),
            "design_result": ("design", lambda c, ts: f"任务：\n{ts}\n设计结果：\n{c}", "cache_design"),
            "task_update": ("design", lambda c, ts: f"任务更新：\n{c}", "cache_full_plan"),
            "search_result": ("search", lambda c, ts: f"任务：\n{ts}\n搜索结果：\n{c}", "cache_design"),
        }
        
        if content_type not in content_configs:
            raise ValueError(f"不支持的内容类型: {content_type}")
        
        category, format_func, cache_attr = content_configs[content_type]
        mem0_content = format_func(content, task_str)
        getattr(self, cache_attr).clear()  # 清理对应缓存

        parent_task_id = ".".join(task_id.split(".")[:-1]) if task_id and "." in task_id else ""
        mem_metadata = {
            "category": category,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "hierarchy_level": len(task_id.split(".")),
            "parent_task_id": parent_task_id,
            "dependency": dependency_str,
        }
        
        logger.info(f"mem0 添加记忆 {self.writing_mode}_{self.language}_{hashkey}_{category}\n{mem0_content}\n{mem_metadata}")
        self.client.add(
            mem0_content,
            user_id=f"{self.writing_mode}_{self.language}_{hashkey}_{category}",
            metadata=mem_metadata
        )

    def search(self, hashkey, querys, category, limit=30, filters=None):
        if not querys:
            return []
        
        # 去重并过滤空查询
        unique_queries = list(dict.fromkeys([q.strip() for q in querys if q and q.strip()]))
        if not unique_queries:
            return []
        
        user_id = f"{self.writing_mode}_{self.language}_{hashkey}_{category}"
        
        # 使用 OR 连接所有独立查询，以最大化召回可能相关的文档。
        # 检索的精确性更多地依赖于后续的重排序步骤，而不是复杂的布尔查询。
        query = " OR ".join([f"({q})" for q in unique_queries])
        
        logger.info(f"mem0 搜索记忆 {user_id}\n查询: {query[:200]}...\n查询数: {len(unique_queries)}\n限制: {limit}\n过滤器: {filters}")
        
        results = self.client.search(
            query=query,
            user_id=user_id,
            limit=limit,
            filters=filters
        )
        logger.info(f"mem0 搜索完成: {len(unique_queries)} 个查询返回 {results}")
        return results
    

    def get_outer_graph_dependent(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        task_id = task_info.get("id")
        if not task_id or task_id == "" or task_id == "0":
            raise ValueError(f"任务信息中未找到任务ID {task_id} \n 任务信息: {task_info}")

        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        if cur_hierarchy_level <= 2:
            return ""
        
        s = f"{hashkey}\n{task_info}\n{same_graph_dependent_designs}\n{latest_content}"
        cache_key = hashlib.blake2b(s.encode('utf-8'), digest_size=32).hexdigest()
        cached_result = self.cache_design.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 写作模式方法映射
        method_map = {
            "story": lambda: self.get_story_outer_graph_dependent(hashkey, task_info, same_graph_dependent_designs, latest_content),
            "book": lambda: "",
            "report": lambda: ""
        }
        
        if self.writing_mode not in method_map:
            raise ValueError(f"不支持的写作模式: {self.writing_mode}")
        
        ret = method_map[self.writing_mode]()
        self.cache_design.set(cache_key, ret)
        return ret

    def get_content(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        task_id = task_info.get("id")
        if not task_id or task_id == "" or task_id == "0":
            raise ValueError(f"任务信息中未找到任务ID {task_id} \n 任务信息: {task_info}")

        if not latest_content.strip() or latest_content.strip() == "" or  len(latest_content) < 500:
            return ""

        s = f"{hashkey}\n{task_info}\n{same_graph_dependent_designs}\n{latest_content}"
        cache_key = hashlib.blake2b(s.encode('utf-8'), digest_size=32).hexdigest()
        cached_result = self.cache_text.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 写作模式方法映射
        method_map = {
            "story": lambda: self.get_story_content(hashkey, task_info, same_graph_dependent_designs, latest_content),
            "book": lambda: "",
            "report": lambda: ""
        }
        
        if self.writing_mode not in method_map:
            raise ValueError(f"不支持的写作模式: {self.writing_mode}")
        
        ret = method_map[self.writing_mode]()
        self.cache_text.set(cache_key, ret)
        return ret

    def _generate_queries(self, category, task_info, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索"设计库"、"小说正文"的搜索查询词。
        """
        if category not in ["design", "text"]:
            raise ValueError(f"不支持的类别: {category}")
        
        # 配置映射：{(category, writing_mode, language): (prompt_system, prompt_user_template)}
        prompt_config = {
            ("design", "story", "zh"): (mem_story_design_queries_zh_system, mem_story_design_queries_zh_user),
            ("text", "story", "zh"): (mem_story_text_queries_zh_system, mem_story_text_queries_zh_user),
            ("design", "report", "zh"): (mem_report_design_queries_zh_system, mem_report_design_queries_zh_user),
            ("text", "report", "zh"): (mem_report_text_queries_zh_system, mem_report_text_queries_zh_user),
        }
        config_key = (category, self.writing_mode, self.language)
        if config_key not in prompt_config:
            raise ValueError(f"未配置提示词: {config_key}")
        
        prompt_system, prompt_user_template = prompt_config[config_key]
        prompt_user = prompt_user_template.format(
            task_info=json.dumps(task_info, ensure_ascii=False, indent=2),
            context_str=context_str
        )

        llm_call_func = {
            "zh": llm_client.call_fast_zh,
            "en": llm_client.call_fast_en
        }.get(self.language)
        if not llm_call_func:
            raise ValueError(f"不支持的语言: {self.language}")
        
        response = llm_call_func(
            messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}],
            temperature=0.2,
        )
        if response is None:
            raise ValueError("LLM响应为空")
        
        content = response[0].message.content
        logger.info(f"mem0 生成查询 {self.writing_mode}_{self.language}_{category}\n响应: {content}")
        
        queries = json.loads(content)
        if not isinstance(queries, list):
            raise ValueError(f"无效的查询: {queries}")
        
        return queries

    def get_story_outer_graph_dependent(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        """
        检索设计结果
        替换 agent/agents/regular.py 中的 get_llm_output 中的 to_run_outer_graph_dependent
        """
        llm_queries = self._generate_queries(
            category="design",
            task_info=task_info,
            context_str=f"相关设计:\n{same_graph_dependent_designs}\n\n最新内容:\n{latest_content}"
        )
        
        task_id = task_info.get('id', '')
        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        filters = {
            "content_type": "design_result",
            "hierarchy_level": {"gte": 1, "lt": cur_hierarchy_level}
        }
        all_results = self.search(hashkey, llm_queries, "design", limit=500, filters=filters)
        
        # 简化的排序策略：主要基于相关性和时间新鲜度
        def sort_key_design(x):
            meta = x.get('metadata', {})
            base_score = x.get('score', 0.0)  # mem0向量检索的相关性分数
            
            # 解析时间戳 - 时间新鲜度很重要
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
                time_diff_hours = (datetime.now() - timestamp).total_seconds() / 3600
                # 时间衰减：越新的内容越重要，但不要过度复杂化
                time_factor = max(0.1, 1.0 - time_diff_hours / (24 * 7))  # 一周内线性衰减
            except (ValueError, TypeError):
                time_factor = 0.1
            
            # 简化的综合评分：相关性为主，时间为辅
            final_score = base_score * (0.8 + 0.2 * time_factor)
            return final_score
        
        sorted_results = sorted(all_results, key=sort_key_design, reverse=True)
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"mem0 获取故事外部图依赖 {self.writing_mode}_{self.language}_{hashkey}\n{task_info}\n{combined_results}")
        return "\n\n".join(combined_results)

    def get_story_content(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        """
        检索小说已经写的正文内容
        替换 agent/agents/regular.py 中的 get_llm_output 中的 memory.article
        """
        llm_queries = self._generate_queries(
            category="text",
            task_info=task_info,
            context_str=f"最新正文内容:\n{latest_content}\n\n相关设计:\n{same_graph_dependent_designs}"
        )
        all_results = self.search(hashkey, llm_queries, "text", limit=500)
        
        # 简化的文本排序策略：时间优先，相关性为辅
        def sort_key_text(x):
            meta = x.get('metadata', {})
            base_score = x.get('score', 0.0)
            
            # 时间新鲜度：对于小说内容，时间顺序很重要
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
                # 直接使用时间戳作为主要排序依据，最新的内容排在前面
                time_score = timestamp.timestamp() 
            except (ValueError, TypeError):
                time_score = 0
            
            # 结合相关性分数，但时间权重更高
            final_score = time_score + base_score * 100  # 时间为主，相关性为辅
            return final_score
        
        sorted_results = sorted(all_results, key=sort_key_text, reverse=True)
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"mem0 获取故事内容 {self.writing_mode}_{self.language}_{hashkey}\n{task_info}\n{combined_results}")
        return "\n\n".join(combined_results)

    def get_full_plan(self, hashkey, task_info):
        pass
 

###############################################################################


_mem0_cache = {}


def get_mem0(config):
    writing_mode = config["writing_mode"]
    language = config["language"]
    
    if writing_mode not in ["story", "book", "report"]:
        raise ValueError(f"不支持的写作模式: {writing_mode}")
    
    if language not in ["zh", "en"]:
        raise ValueError(f"不支持的语言: {language}")
    
    cache_key = f"{writing_mode}_{language}"
    if cache_key not in _mem0_cache:
        _mem0_cache[cache_key] = Mem0(writing_mode, language)
    
    return _mem0_cache[cache_key]
