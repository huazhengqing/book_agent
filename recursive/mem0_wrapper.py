#coding: utf8
import os
from loguru import logger
from recursive.llm.litellm_proxy import llm_client
import json
from mem0 import Memory as Mem0Memory
from datetime import datetime
from recursive.utils.keyword_extractor_zh import keyword_extractor_zh
from recursive.utils.keyword_extractor_en import keyword_extractor_en
from recursive.agent.prompts.story_zh.mem import (
    mem_story_fact_zh,
    mem_story_update_zh,
    mem_story_design_queries_zh_system,
    mem_story_design_queries_zh_user,
    mem_story_text_queries_zh_system,
    mem_story_text_queries_zh_user
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
分析 mem0_wrapper.py，撰写一份全面的分析报告，检查是否存在逻辑不一致之处，指出可以改进的地方，如何确保它们更好地协同？


"""

class Mem0:
    def __init__(self, writing_mode, language):
        # 确定写作模式（story, book, report）
        self.writing_mode = writing_mode
        self.language = language
        self.keyword_extractor = None 
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
        if self.writing_mode == "story":
            self.mem0_config["custom_fact_extraction_prompt"] = mem_story_fact_zh
            self.mem0_config["custom_update_memory_prompt"] = mem_story_update_zh
        elif self.writing_mode == "book":
            self.mem0_config["custom_fact_extraction_prompt"] = "" # TODO: Add book-specific fact extraction prompt
            self.mem0_config["custom_update_memory_prompt"] = "" # TODO: Add book-specific memory update prompt
        elif self.writing_mode == "report":
            self.mem0_config["custom_fact_extraction_prompt"] = "" # TODO: Add report-specific fact extraction prompt
            self.mem0_config["custom_update_memory_prompt"] = "" # TODO: Add report-specific memory update prompt
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")

        self.client = None

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cache_dir = os.path.join(project_root, ".cache", "mem0_text")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_text = diskcache.Cache(cache_dir, size_limit=1024 * 1024 * 300)

        cache_dir = os.path.join(project_root, ".cache", "mem0_design")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_design = diskcache.Cache(cache_dir, size_limit=1024 * 1024 * 300)
        
    def get_client(self):
        if not self.client:
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
        return self.client

    def get_keyword_extractor(self):
        if not self.keyword_extractor:
            if self.language == "zh":
                self.keyword_extractor = keyword_extractor_zh
            elif self.language == "en":
                self.keyword_extractor = keyword_extractor_en
            else:
                raise ValueError(f"Unsupported language: {self.language}")
        return self.keyword_extractor

    def add(self, hashkey, content, content_type, task_info):
        task_id = task_info.get("id")
        if not task_id:
            raise ValueError("Task ID not found in task_info")
        task_type = task_info.get("task_type")
        task_goal = task_info.get("goal")
        dependency = task_info.get("dependency")
        task_str = json.dumps(task_info, ensure_ascii=False)

        if content_type == "text_content":
            category = "text"
        elif content_type in ["task_update", "task_decomposition", "design_result"]:
            category = "design"

        mem0_content = ""
        if content_type == "text_content":
            mem0_content = content
        elif content_type == "task_decomposition":
            if isinstance(content, dict):
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)
            mem0_content = f"""任务：\n{task_str}\n规划分解结果：\n{content_str}"""
        elif content_type == "design_result":
            mem0_content = f"""任务：\n{task_str}\n设计结果：\n{content}"""
        elif content_type == "task_update":
            if isinstance(content, dict):
                content_str = json.dumps(content, ensure_ascii=False)
            else:
                content_str = str(content)
            mem0_content = f"""任务更新：\n{content_str}"""

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
            "content_hash": hashlib.md5(mem0_content.encode('utf-8')).hexdigest()
        }
        
        logger.info(f"mem0 add() {self.writing_mode}_{self.language}_{hashkey}_{category}\n{mem0_content}\n{mem_metadata}")
        self.get_client().add(
            mem0_content,
            user_id=f"{self.writing_mode}_{self.language}_{hashkey}_{category}",
            metadata=mem_metadata
        )
        
        if category == "design":
            self.cache_design.clear()
        if category == "text":
            self.cache_text.clear()

    def search(self, hashkey, querys, category, limit=1, filters=None):
        query = " ".join(querys)
        logger.info(f"mem0 search() {self.writing_mode}_{self.language}_{hashkey}_{category}\n{query}\n{limit}\n{filters}")
        results = self.get_client().search(
            query=query,
            user_id=f"{self.writing_mode}_{self.language}_{hashkey}_{category}",
            limit=limit,
            filters=filters
        )
        logger.info(f"mem0 search() {self.writing_mode}_{self.language}_{hashkey}_{category}\n{results}")
        return results
        
    def get_outer_graph_dependent(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        s = f"{hashkey}\n{task_info}\n{same_graph_dependent_designs}\n{latest_content}"
        cache_key = hashlib.blake2b(s.encode('utf-8'), digest_size=32).hexdigest()
        cached_result = self.cache_design.get(cache_key)
        if cached_result is not None:
            return cached_result

        ret = ""
        if self.writing_mode == "story":
            ret = self.get_story_outer_graph_dependent(hashkey, task_info, same_graph_dependent_designs, latest_content)
        elif self.writing_mode == "book":
            ret = ""
        elif self.writing_mode == "report":
            ret = ""
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")
        self.cache_design.set(cache_key, ret)
        return ret

    def get_content(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        s = f"{hashkey}\n{task_info}\n{same_graph_dependent_designs}\n{latest_content}"
        cache_key = hashlib.blake2b(s.encode('utf-8'), digest_size=32).hexdigest()
        cached_result = self.cache_text.get(cache_key)
        if cached_result is not None:
            return cached_result

        ret = ""
        if self.writing_mode == "story":
            ret = self.get_story_content(hashkey, task_info, same_graph_dependent_designs, latest_content)
        elif self.writing_mode == "book":
            ret = ""
        elif self.writing_mode == "report":
            ret = ""
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")
        self.cache_text.set(cache_key, ret)
        return ret

    def _generate_queries(self, category, task_goal, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索“设计库”、“小说正文”的搜索查询词。
        """
        prompt_system = ""
        prompt_user = ""
        if category == "design":
            if self.writing_mode == "story":
                if self.language == "zh":
                    prompt_system = mem_story_design_queries_zh_system
                    prompt_user = mem_story_design_queries_zh_user.format(task_goal=task_goal, context_str=context_str)
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            elif self.writing_mode == "book":
                if self.language == "zh":
                    prompt_system = ""
                    prompt_user = ""
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            elif self.writing_mode == "report":
                if self.language == "zh":
                    prompt_system = ""
                    prompt_user = ""
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            else:
                raise ValueError(f"writing_mode={self.writing_mode} not supported")
        elif category == "text":
            if self.writing_mode == "story":
                if self.language == "zh":
                    prompt_system = mem_story_text_queries_zh_system
                    prompt_user = mem_story_text_queries_zh_user.format(task_goal=task_goal, context_str=context_str)
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            elif self.writing_mode == "book":
                if self.language == "zh":
                    prompt_system = ""
                    prompt_user = ""
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            elif self.writing_mode == "report":
                if self.language == "zh":
                    prompt_system = ""
                    prompt_user = ""
                elif self.language == "en":
                    prompt_system = ""
                    prompt_user = ""
                else:
                    raise ValueError(f"Unsupported language: {self.language}")
            else:
                raise ValueError(f"writing_mode={self.writing_mode} not supported")
        else:
            raise ValueError(f"category={category} not supported")
        logger.info(f"mem0 _generate_queries() {self.writing_mode}_{self.language}_{category}\n{task_goal}\n{prompt}")
        response = None
        if self.language == "zh":
            response = llm_client.call_fast_zh(
                messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}],
                temperature=0.2,
            )
        elif self.language == "en":
            response = llm_client.call_fast_en(
                messages=[{"role": "system", "content": prompt_system}, {"role": "user", "content": prompt_user}],
                temperature=0.2,
            )
        else:
            raise ValueError(f"Unsupported language: {self.language}")
        if response == None:
            raise ValueError("LLM response is None")
        content = response[0].message.content
        logger.info(f"mem0 _generate_queries() {self.writing_mode}_{self.language}_{category}\n{task_goal}\n{content}")
        try:
            queries = json.loads(content)
            if isinstance(queries, list):
                logger.info(f"mem0 _generate_queries() {self.writing_mode}_{self.language}_{category}\n{task_goal}\n{queries}")
                return queries
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM for design queries. Content: {content}")
        return []

    def get_story_outer_graph_dependent(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        """
        检索设计结果
        替换 agent/agents/regular.py 中的 get_llm_output 中的 to_run_outer_graph_dependent
        当任务 是write任务时，必需要检索 叙事风格: 确定叙事视角、语言风格、文笔基调、核心叙事策略（如展示/讲述比例）
        工作流程：
            1. 动态查询生成 ：通过 _generate_design_queries 方法，使用轻量级 LLM 根据任务目标和上下文生成搜索查询词
            2. 关键词增强 ：
                - 将任务目标也作为查询词
                - 从最新内容中提取关键词
                - 从相关设计中提取关键词
            3. 叙事风格检索（针对写作任务）：
                - 检索叙事视角设定
                - 检索语言风格指南
                - 检索文笔基调规范
                - 检索核心叙事策略（如展示/讲述比例）
            4. 内容检索 ：调用 search 方法从向量数据库中检索相关设计内容
            5. 结果优化 ：
                - 按层级（越高越重要）、相关度评分、内容长度和时间戳排序
                - 去重处理，避免重复内容
        """
        task_id = task_info.get('id', '')
        if not task_id:
            raise ValueError("Task ID not found in task_info")
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        
        # 使用LLM动态生成查询 
        all_queries = self._generate_queries(
            category="design",
            task_goal=task_goal,
            context_str=f"相关设计:\n{same_graph_dependent_designs}\n\n最新内容:\n{latest_content}"
        )

        all_queries.append(task_goal)
        
        # 针对写作任务，添加叙事风格相关的必需查询
        if task_type and 'write' in task_type.lower():
            narrative_style_queries = [
                "叙事风格",
                "叙事视角 人称设定 POV",
                "语言风格 文笔特色 措辞风格", 
                "文笔基调 情感基调 氛围设定",
                "叙事策略 展示比例 讲述比例",
                "叙事技巧 描写手法 表现手法",
                "章节风格 段落风格 句式特点",
                "对话风格 内心独白 心理描写",
                "节奏控制 张弛有度 情节节奏"
            ]
            all_queries.extend(narrative_style_queries)

        combined_content = f"{latest_content}\n\n{same_graph_dependent_designs}"
        if combined_content:
            all_queries.extend(self.get_keyword_extractor().extract_from_markdown(combined_content))

        final_queries = list(dict.fromkeys(all_queries))
        
        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        filters = {
            "content_type": "design_result",
            "hierarchy_level": {"gte": 1, "lte": cur_hierarchy_level}
        }
        all_results = self.search(hashkey, final_queries, "design", limit=500, filters=filters)
        
        # 按重要性和层级排序，优化排序策略。使用 reverse=True 替代对数值和字符串使用负号。
        def sort_key_design(x):
            meta = x.get('metadata', {})
            # 解析时间戳，如果不存在或格式错误，则使用一个很早的时间
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
            except (ValueError, TypeError):
                timestamp = datetime.min
            return meta.get('hierarchy_level', 0), x.get('score', 0), meta.get('content_length', 0), timestamp
        sorted_results = sorted(all_results, key=sort_key_design, reverse=True)
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"mem0 get_story_outer_graph_dependent() {self.writing_mode}_{self.language}_{hashkey}\n{task_goal}\n{combined_results}")
        return "\n\n".join(combined_results)

    def get_story_content(self, hashkey, task_info, same_graph_dependent_designs, latest_content):
        """
        检索小说已经写的正文内容
        替换 agent/agents/regular.py 中的 get_llm_output 中的 memory.article
        """
        task_id = task_info.get('id', '') 
        if not task_id:
            raise ValueError("Task ID not found in task_info")
        task_goal = task_info.get('goal', '')
        
        # 使用LLM动态生成查询
        all_queries = self._generate_queries(
            category="text",
            task_goal=task_goal,
            context_str=f"最新正文内容:\n{latest_content}\n\n相关设计:\n{same_graph_dependent_designs}"
        )

        all_queries.append(task_goal)

        combined_content = f"{latest_content}\n\n{same_graph_dependent_designs}"
        if combined_content:
            all_queries.extend(self.get_keyword_extractor().extract_from_markdown(combined_content))

        final_queries = list(dict.fromkeys(all_queries))
        
        all_results = self.search(hashkey, final_queries, "text", limit=500)
        
        # 按时间戳和相关性排序，优先最新且相关的内容。使用 reverse=True。
        def sort_key_text(x):
            meta = x.get('metadata', {})
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
            except (ValueError, TypeError):
                timestamp = datetime.min
            return timestamp, x.get('score', 0), meta.get('hierarchy_level', 0), meta.get('content_length', 0)
        sorted_results = sorted(all_results, key=sort_key_text, reverse=True)
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"mem0 get_story_content() {self.writing_mode}_{self.language}_{hashkey}\n{task_goal}\n{combined_results}")
        return "\n\n".join(combined_results)

    def get_full_plan(self, hashkey, task_info):
        task_id = task_info.get("id", "")
        if not task_id:
            raise ValueError("Task ID not found in task_info")
        task_goal = task_info.get('goal', '')
        
        to_task_ids = []
        current_id = task_id
        to_task_ids.append(current_id)
        while "." in current_id:
            current_id = ".".join(current_id.split(".")[:-1])
            to_task_ids.append(current_id)
        to_task_ids = sorted(to_task_ids, key=lambda x: len(x.split(".")))

        task_goals = []
        for pid in to_task_ids:
            results = self.get_client().search(
                query=f"任务id为{pid}的详细目标(goal)",
                user_id=f"{self.writing_mode}_{self.language}_{hashkey}_design",
                limit=1
            )
            results = results if isinstance(results, list) else results.get('results', [])
            if not results:
                continue
            task_goals.append(f"[{pid}]: {results[0].get('memory')}")
        logger.info(f"mem0 get_story_content() {self.writing_mode}_{self.language}_{hashkey}\n{task_goal}\n{task_goals}")
        return "\n".join(task_goals)


###############################################################################


mem0_story_zh = Mem0("story", "zh")
mem0_story_en = Mem0("story", "en")
mem0_book_zh = Mem0("book", "zh")
mem0_book_en = Mem0("book", "en")
mem0_report_zh = Mem0("report", "zh")
mem0_report_en = Mem0("report", "en")


def get_mem0(config):
    if config["writing_mode"] == "story":
        if config["language"] == "zh":
            return mem0_story_zh
        elif config["language"] == "en":
            return mem0_story_en
        else:
            raise ValueError(f"Unsupported language: {config["language"]}")
    elif config["writing_mode"] == "book":
        if config["language"] == "zh":
            return mem0_book_zh
        elif config["language"] == "en":
            return mem0_book_en
        else:
            raise ValueError(f"Unsupported language: {config["language"]}")
    elif config["writing_mode"] == "report":
        if config["language"] == "zh":
            return mem0_report_zh
        elif config["language"] == "en":
            return mem0_report_en
        else:
            raise ValueError(f"Unsupported language: {config["language"]}")
    else:
        raise ValueError(f"writing_mode={config["writing_mode"]} not supported")
