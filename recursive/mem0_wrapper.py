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
from recursive.utils.keyword_extractor_zh import keyword_extractor_zh
from recursive.utils.keyword_extractor_en import keyword_extractor_en
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
            self.keyword_extractor = keyword_extractor_zh
        elif self.language == "en":
            self.keyword_extractor = keyword_extractor_en
        else:
            raise ValueError(f"Unsupported language: {self.language}")
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
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")
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
        elif self.writing_mode == "book":
            return ""
        elif self.writing_mode == "report":
            return ""
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")



    def get_content(self, task_info, same_graph_dependent_designs, latest_content):
        if self.writing_mode == "story":
            return self.get_story_content(task_info, same_graph_dependent_designs, latest_content)
        elif self.writing_mode == "book":
            return ""
        elif self.writing_mode == "report":
            return ""
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")







    def _generate_design_queries(self, task_goal, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索“设计库”的搜索查询词。
        """
        prompt = ""
        if self.writing_mode == "story":
            prompt = mem_story_design_queries.format(task_goal=task_goal, context_str=context_str)
        elif self.writing_mode == "book":
            pass
        elif self.writing_mode == "report":
            pass
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")
        response = self.llm_client.call_fast(
            model=self.fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response[0].message.content
        try:
            queries = json.loads(content)
            if isinstance(queries, list):
                logger.info(f"_generate_design_queries(): {queries}")
                return queries
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM for design queries. Content: {content}")
        return []


    def _generate_text_queries(self, task_goal, context_str):
        """
        使用轻量级LLM根据任务和上下文动态生成用于检索“正文库”的搜索查询词。
        """
        prompt = ""
        if self.writing_mode == "story":
            prompt = mem_story_text_queries.format(task_goal=task_goal, context_str=context_str)
        elif self.writing_mode == "book":
            pass
        elif self.writing_mode == "report":
            pass
        else:
            raise ValueError(f"writing_mode={self.writing_mode} not supported")
        response = self.llm_client.call_fast(
            model=self.fast_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        content = response[0].message.content
        try:
            queries = json.loads(content)
            if isinstance(queries, list):
                logger.info(f"_generate_text_queries(): {queries}")
                return queries
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from LLM for text queries. Content: {content}")
        return []
    



    def get_story_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 to_run_outer_graph_dependent
        """
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        
        # 使用LLM动态生成查询 
        all_queries = self._generate_design_queries(
            task_goal=task_goal,
            context_str=f"相关设计:\n{same_graph_dependent_designs}\n\n最新内容:\n{latest_content}"
        )

        all_queries.append(task_goal)

        if latest_content:
            all_queries.extend(self.keyword_extractor.extract_from_text(latest_content))

        if same_graph_dependent_designs:
            all_queries.extend(self.keyword_extractor.extract_from_markdown(same_graph_dependent_designs))

        final_queries = list(dict.fromkeys(all_queries))
        
        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        filters = {
            "content_type": "design_result",
            "hierarchy_level": {"gte": 1, "lte": cur_hierarchy_level}
        }
        all_results = self.search(final_queries, "design", limit=500, filters=filters)
        
        
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
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"get_story_outer_graph_dependent() combined_results=\n{combined_results}")
        return "\n\n".join(combined_results)





    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 memory.article
        """
        task_goal = task_info.get('goal', '')
        task_id = task_info.get('id', '') 
        
        # 使用LLM动态生成查询
        all_queries = self._generate_text_queries(
            task_goal=task_goal,
            context_str=f"最新正文内容:\n{latest_content}\n\n相关设计:\n{same_graph_dependent_designs}"
        )

        all_queries.append(task_goal)

        if latest_content:
            all_queries.extend(self.keyword_extractor.extract_from_text(latest_content))

        if same_graph_dependent_designs:
            all_queries.extend(self.keyword_extractor.extract_from_markdown(same_graph_dependent_designs))

        final_queries = list(dict.fromkeys(all_queries))
        
        all_results = self.search(final_queries, "text", limit=500)
        
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
        
        combined_results = []
        seen_contents = set()
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                combined_results.append(memory_content)
                seen_contents.add(memory_content)
        logger.info(f"get_story_content() combined_results=\n{combined_results}")
        return "\n\n".join(combined_results)





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






