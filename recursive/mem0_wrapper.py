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
from recursive.utils.keyword_extractor_zh import KeywordExtractorZh
from recursive.utils.keyword_extractor_en import KeywordExtractorEn
from recursive.agent.prompts.story_zh.mem import (
    mem_story_fact,
    mem_story_update,
    mem_story_design_queries,
    mem_story_text_queries
)


class Mem0:
    def __init__(self, root_node, config):
        # 确定写作模式（story, book, report）
        self.writing_mode = config.get("writing_mode", "story")
        self.language = config.get("language", "zh")
        self.config = config
        self.root_node = root_node
        
        self.user_id_pre = f"{self.writing_mode}_{self.root_node.hashkey}"

        if self.language == "zh":
            self.keyword_extractor = KeywordExtractorZh(mode=self.writing_mode)
        else:
            self.keyword_extractor = KeywordExtractorEn(mode=self.writing_mode)

        self.llm_client = LiteLLMProxy()
        self.fast_model = os.environ.get("fast_model")

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
            # "history_db_path": "./.mem0/history.db"
        }
        if self.writing_mode == "story":
            self.config["custom_fact_extraction_prompt"] = mem_story_fact
            self.config["custom_update_memory_prompt"] = mem_story_update
        elif self.writing_mode == "book":
            self.config["custom_fact_extraction_prompt"] = ""
            self.config["custom_update_memory_prompt"] = ""
        elif self.writing_mode == "report":
            self.config["custom_fact_extraction_prompt"] = ""
            self.config["custom_update_memory_prompt"] = ""
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
        




    def add(self, content, content_type, task_info):
        task_id = task_info.get("id")
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
        
        logger.info(f"mem0 add() mem0_content=\n{mem0_content}\n mem_metadata=\n{mem_metadata}")
        self.client.add(
            mem0_content,
            user_id=f"{self.user_id_pre}_{category}",
            metadata=mem_metadata
        )

    def search(self, querys, category, limit=1, filters=None):
        query = " ".join(querys)
        logger.info(f"mem0 search() {category} query=\n{query}")
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_{category}",
            limit=limit,
            filters=filters
        )
        logger.info(f"mem0 search() results=\n{results}")
        return results
        


    def get_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        if self.writing_mode == "story":
            return self.get_story_outer_graph_dependent(task_info, same_graph_dependent_designs, latest_content)



    def get_content(self, task_info, same_graph_dependent_designs, latest_content):
        if self.writing_mode == "story":
            return self.get_story_content(task_info, same_graph_dependent_designs, latest_content)







    def _generate_design_queries(self, task_goal, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索“设计库”的搜索查询词。
        """
        prompt = ""
        if self.writing_mode == "story":
            prompt = mem_story_design_queries.format(task_goal=task_goal, context_str=context_str)
        response = self.llm_client.call_fast(
            model=self.fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response[0].message.content
        queries = json.loads(content)
        if isinstance(queries, list):
            logger.info(f"动态生成的'设计'查询词: {queries}")
            return queries
        return []


    def _generate_text_queries(self, task_goal, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索“正文库”的搜索查询词。
        """
        prompt = ""
        if self.writing_mode == "story":
            prompt = mem_story_text_queries.format(task_goal=task_goal, context_str=context_str)
        response = self.llm_client.call_fast(
            model=self.fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response[0].message.content
        queries = json.loads(content)
        if isinstance(queries, list):
            logger.info(f"动态生成的'正文'查询词: {queries}")
            return queries
        return []
    



    def get_story_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 to_run_outer_graph_dependent
        """
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        
        # 1. 使用LLM动态生成查询 (新功能)
        llm_queries = self._generate_design_queries(
            task_goal=task_goal,
            context_str=f"相关设计:\n{same_graph_dependent_designs}\n\n最新内容:\n{latest_content}"
        )
        all_queries = llm_queries

        # 2. 基于规则的查询生成（作为补充和保底）
        all_queries.append(task_goal)

        # 3. 层级信息提取 - 基于新标签体系
        level_patterns = [
            (r'第(\d+)卷', '卷', '#第{}卷'),
            (r'第(\d+)幕', '幕', '#第{}幕'),
            (r'第(\d+)章', '章', '#第{}章'),
            (r'场景(\d+)', '场景', '#场景{}'),
            (r'节拍(\d+)', '节拍', '#节拍{}')
        ]
        for pattern, level_name, tag_template in level_patterns:
            matches = re.findall(pattern, task_goal)
            for num in matches:
                # 精确层级匹配
                all_queries.append(f"第{num}{level_name}")
                all_queries.append(tag_template.format(num))
                # 上级层级上下文（获取更高层级的设计指导）
                try:
                    num_int = int(num)
                    if level_name == '章' and num_int > 1:
                        all_queries.append(f"第{num_int-1}章")
                        all_queries.append(f"#第{num_int-1}章")
                    elif level_name == '幕':
                        all_queries.append(f"#幕级")
                    elif level_name == '卷':
                        all_queries.append(f"#卷级")
                except ValueError:
                    logger.warning(f"Invalid number format for level: {num}")
        
        # 4. 任务ID层级关联 - 检索父任务的设计结果
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id}")
            # 添加层级标签
            hierarchy_level = len(task_id.split('.'))
            if hierarchy_level <= 2:
                all_queries.append("#全书级")
            elif hierarchy_level <= 3:
                all_queries.append("#卷级")
            elif hierarchy_level <= 4:
                all_queries.append("#幕级")
            elif hierarchy_level <= 5:
                all_queries.append("#场景级")

        # 5. 从同层设计方案提取关键词 (强信号)
        if same_graph_dependent_designs:
            design_keywords = self.keyword_extractor.extract_from_markdown(same_graph_dependent_designs)
            important_keywords = [kw for kw in design_keywords[:20] if len(kw) > 1]
            all_queries.extend(important_keywords)

        # 去重并优化查询词
        final_queries = list(dict.fromkeys(all_queries))
        logger.info(f"Design search queries: {final_queries[:10]}...")  # 记录部分查询词
        
        # 构建过滤条件
        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        filters = {
            "content_type": "design_result",
            "hierarchy_level": {"gte": 1, "lte": cur_hierarchy_level}
        }
        all_results = self.search(final_queries, "design", limit=50, filters=filters)
        
        combined_results = []
        seen_contents = set()
        
        # 按重要性和层级排序，优化排序策略
        sorted_results = sorted(all_results, key=lambda x: (
            # 优先级1：层级越高越重要
            -x.get('metadata', {}).get('hierarchy_level', 0),
            # 优先级2：与当前任务的相关度评分
            -x.get('score', 0),
            # 优先级3：内容长度（更详细的设计）
            -x.get('metadata', {}).get('content_length', 0),
            # 优先级4：时间戳（更新的设计）
            -x.get('metadata', {}).get('timestamp', '')
        ))
        
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                # 过滤条件：内容长度和质量
                if len(memory_content) > 50 and any(keyword in memory_content for keyword in ['设计方案', '设定', '规划', '架构']):
                    combined_results.append(memory_content)
                    seen_contents.add(memory_content)
                # 限制最终结果数量，避免上下文过长。原值为2000，过高，可能导致性能问题和上下文超限。
                # 调整为更合理的值，例如20个高质量的设计文档。
                if len(combined_results) >= 20:
                    break
        
        logger.info(f"Found {len(combined_results)} relevant design results")
        return "\n\n".join(combined_results)

    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 memory.article
        """
        task_goal = task_info.get('goal', '')
        task_id = task_info.get('id', '') 
        
        # 1. 使用LLM动态生成查询 (新功能)
        llm_queries = self._generate_text_queries(
            task_goal=task_goal,
            context_str=f"最新正文内容:\n{latest_content}"
        )
        all_queries = llm_queries

        # 2. 基于规则的查询生成（作为补充和保底）
        all_queries.append(task_goal)

        # 3. 当前章节上下文 - 将"最新"、"近期"转化为具体章节检索词
        current_chapter_queries = self._get_current_chapter_context(task_goal, task_id)
        all_queries.extend(current_chapter_queries)
        
        # 4. 任务ID关联 - 检索相关任务的写作内容
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id}")
            all_queries.append(f"ID: {parent_id}")
            # 添加层级标签
            hierarchy_level = len(task_id.split('.'))
            if hierarchy_level >= 3:
                all_queries.append("#章级")
            elif hierarchy_level >= 4:
                all_queries.append("#场景级")
            elif hierarchy_level >= 5:
                all_queries.append("#节拍级")
        
        # 6. 从最新内容提取关键词 - 优先考虑最新内容
        if latest_content and isinstance(latest_content, str):
            content_keywords = self.keyword_extractor.extract_from_text(latest_content, top_k=25)
            # 优先添加最新内容的关键词，放在检索词列表最前面
            all_queries = content_keywords + all_queries
            logger.info(f"Latest content keywords: {content_keywords[:10]}...")  # 记录部分关键词

        # 8. 去重并优化查询词
        final_queries = list(dict.fromkeys(all_queries))
        logger.info(f"Content search queries: {final_queries[:10]}...")  # 记录部分查询词
        
        # 8. 执行检索 - 检索正文内容，使用更精准的过滤条件
        all_results = self.search(final_queries, "text", limit=50)
        
        # 9. 结果处理和排序 - 优化排序策略
        combined_results = []
        seen_contents = set()
        
        # 按时间戳和相关性排序，优先最新且相关的内容
        sorted_results = sorted(all_results, key=lambda x: (
            # 优先级1：时间戳（最新内容）倒序排列
            -x.get('metadata', {}).get('timestamp', ''),
            # 优先级2：与当前任务的相关度评分
            -x.get('score', 0),
            # 优先级3：层级（更具体的内容）
            -x.get('metadata', {}).get('hierarchy_level', 0),
            # 优先级4：内容长度（更详细的内容）
            -x.get('metadata', {}).get('content_length', 0)
        ))
        
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                # 过滤条件：确保是正文内容，包含关键标识符
                content_markers = ["#正文内容", "正文内容", "场景", "对话", "情节", "主角", "角色", "剧情", "故事", "金手指"]
                if len(memory_content) > 30 and any(marker in memory_content for marker in content_markers):
                    combined_results.append(memory_content)
                    seen_contents.add(memory_content)
                # 限制最终结果数量，确保获取足够的上下文，同时避免上下文过长。
                # 原值为50，可以根据需要调整，例如减少到30。
                if len(combined_results) >= 30:
                    break
        
        logger.info(f"Found {len(combined_results)} relevant content results")
        return "\n\n".join(combined_results)


    def _get_current_chapter_context(self, task_goal, task_id):
        """
        获取当前章节上下文，将"最新"、"近期"等时间描述词转化为具体的章节检索词
        增强版：增加对"金手指"相关内容的检索支持
        """
        current_chapter_queries = []
        
        # 从任务目标中提取当前章节信息
        level_patterns = [
            (r'第(\d+)卷', '卷'),
            (r'第(\d+)幕', '幕'), 
            (r'第(\d+)章', '章'),
            (r'场景(\d+)', '场景'),
            (r'节拍(\d+)', '节拍')
        ]
        
        current_levels = {}
        for pattern, level_name in level_patterns:
            matches = re.findall(pattern, task_goal)
            if matches:
                try:
                    current_levels[level_name] = int(matches[-1])  # 取最后一个匹配的数字
                except ValueError:
                    logger.warning(f"Invalid number format for {level_name}: {matches[-1]}")
        
        # 基于任务ID推断当前层级
        if task_id and '.' in task_id:
            hierarchy_level = len(task_id.split('.'))
            if hierarchy_level >= 3 and '章' not in current_levels:
                # 尝试从父任务ID推断章节
                parent_tasks = task_id.split('.')
                if len(parent_tasks) >= 3:
                    try:
                        current_levels['章'] = int(parent_tasks[2])
                    except (ValueError, IndexError):
                        logger.warning(f"Failed to infer chapter from task_id: {task_id}")
        
        # 生成当前章节的检索词
        for level_name, num in current_levels.items():
            current_chapter_queries.append(f"第{num}{level_name}")
            current_chapter_queries.append(f"#{level_name}级")
            current_chapter_queries.append(f"#第{num}{level_name}")
            current_chapter_queries.append(f"第{num}{level_name}金手指")  # 增加金手指相关检索
            
            # 添加前几个章节的内容（获取连续性上下文）
            if level_name == '章' and num > 1:
                for i in range(max(1, num-2), num):  # 前2章
                    current_chapter_queries.append(f"第{i}章")
                    current_chapter_queries.append(f"#第{i}章")
                    current_chapter_queries.append(f"第{i}章金手指")  # 增加金手指相关检索
            elif level_name == '幕' and num > 1:
                current_chapter_queries.append(f"第{num-1}幕")
                current_chapter_queries.append(f"#第{num-1}幕")
                current_chapter_queries.append(f"第{num-1}幕金手指")  # 增加金手指相关检索
            elif level_name == '卷' and num > 1:
                current_chapter_queries.append(f"第{num-1}卷")
                current_chapter_queries.append(f"#第{num-1}卷")
                current_chapter_queries.append(f"第{num-1}卷金手指")  # 增加金手指相关检索
        
        # 增加通用金手指检索词
        current_chapter_queries.append("金手指")
        current_chapter_queries.append("#金手指")
        
        return current_chapter_queries


    def get_full_plan(self, task_info):
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
