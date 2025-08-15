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
from recursive.agent.prompts.story_zh.mem import (
    MEM_STORY_FACT,
    MEM_STORY_UPDATE
)


class Mem0:
    def __init__(self, root_node, config):
        # 确定写作模式（story, book, report）
        self.writing_mode = config.get("writing_mode", "story")
        self.root_node = root_node

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




    def search(self, querys, category, limit=1, filters=None):
        query = " ".join(querys)
        logger.info(f"mem0 search() {category} query=\n{query}")
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_{category}",
            limit=limit,
            filters=filters
        )
        results = results if isinstance(results, list) else results.get('results', [])
        contents = []
        for r in results:
            content = r.get('memory', '')
            if content:
                contents.append(content)
        logger.info(f"mem0 search() contents=\n{contents}")
        return contents



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







    def get_story_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        
        final_contents = []

        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        filters={
            "hierarchy_level": {"gte": 1, "lte": cur_hierarchy_level - 1}
        }

        # 基于任务类型的精准检索
        if task_type == "write":
            # 写作任务需要角色状态、情节发展、场景环境
            querys = [
                "角色设计 人物设定 角色弧线",
                "情节设计 剧情结构 悬念布局",
                "世界设计 背景设定 规则体系",
                f"任务目标: {task_goal[:200]}"  # 限制长度
            ]
        else:  # think任务
            # 设计任务需要已有设计框架和规划
            querys = [
                "角色设计 人物设定",
                "情节设计 剧情结构", 
                "世界设计 背景设定",
                f"设计任务: {task_goal[:200]}"
            ]
        
        final_contents.extend(self.search(querys, 'design', 100, filters))
        
        # 处理同层设计信息，避免token超限
        if same_graph_dependent_designs:
            # 提取关键词而非全文
            design_keywords = self._extract_design_keywords(same_graph_dependent_designs)
            if design_keywords:
                querys = [f"相关设计: {design_keywords}"]
                final_contents.extend(self.search(querys, "design", 50, filters))

        return "\n".join(final_contents[:10])  # 限制返回数量








    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')

        final_contents = []

        # 基于任务类型的分层检索策略
        if task_type == "write":
            # 写作任务：重点关注角色状态、情节连贯、场景环境
            final_contents.append("=== 角色当前状态 ===")
            querys = ["角色动态", "角色状态", "人物关系", "角色成长"]
            final_contents.extend(self.search(querys, "story", 50))

            final_contents.append("=== 情节发展脉络 ===") 
            querys = ["情节进展", "事件脉络", "悬念伏笔", "因果关系"]
            final_contents.extend(self.search(querys, "story", 50))

            final_contents.append("=== 场景环境设定 ===")
            querys = ["场景环境", "世界设定", "势力关系", "规则体系"]
            final_contents.extend(self.search(querys, "story", 30))

            # 最新内容关联检索
            if latest_content:
                final_contents.append("=== 最新情节关联 ===")
                latest_keywords = self._extract_content_keywords(latest_content[-500:])
                if latest_keywords:
                    querys = [f"相关情节: {latest_keywords}"]
                    final_contents.extend(self.search(querys, "story", 30))

        else:  # think任务
            # 设计任务：重点关注设计框架、规划思路
            final_contents.append("=== 已有设计框架 ===")
            querys = ["角色设计", "情节设计", "世界设计"]
            final_contents.extend(self.search(querys, "story", 40))

            final_contents.append("=== 相关情节表现 ===")
            querys = ["情节进展", "角色动态", "场景世界"]
            final_contents.extend(self.search(querys, "story", 40))

        # 任务特定信息
        final_contents.append("=== 任务相关信息 ===")
        task_keywords = self._extract_task_keywords(task_goal)
        if task_keywords:
            querys = [f"任务相关: {task_keywords}"]
            final_contents.extend(self.search(querys, "story", 30))

        return "\n".join(final_contents)








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

    def _extract_design_keywords(self, design_text):
        """从设计文本中提取关键词，避免token超限"""
        if not design_text or len(design_text) < 50:
            return ""
        
        # 提取【】标签内的关键词
        import re
        keywords = re.findall(r'【([^】]+)】', design_text[:1000])
        
        # 提取常见的设计关键词
        design_keywords = []
        key_patterns = [
            r'角色.*?设计', r'人物.*?设定', r'主角.*?[：:]([^。\n]+)',
            r'情节.*?设计', r'剧情.*?结构', r'冲突.*?[：:]([^。\n]+)',
            r'世界.*?设计', r'背景.*?设定', r'规则.*?[：:]([^。\n]+)'
        ]
        
        for pattern in key_patterns:
            matches = re.findall(pattern, design_text[:1000])
            design_keywords.extend(matches)
        
        # 合并关键词，限制长度
        all_keywords = keywords + design_keywords
        return " ".join(all_keywords[:10])  # 限制关键词数量

    def _extract_content_keywords(self, content_text):
        """从正文内容中提取关键词"""
        if not content_text:
            return ""
        
        # 提取人名、地名、重要物品等
        import re
        keywords = []
        
        # 提取引号内的对话关键词
        dialogue_matches = re.findall(r'"([^"]{5,30})"', content_text)
        keywords.extend([match[:15] for match in dialogue_matches[:3]])
        
        # 提取动作关键词
        action_patterns = [r'(\w+)(?:走向|冲向|看向|转身|停下)', r'(\w+)(?:说道|喊道|低语)']
        for pattern in action_patterns:
            matches = re.findall(pattern, content_text)
            keywords.extend(matches[:3])
        
        return " ".join(keywords[:8])

    def _extract_task_keywords(self, task_goal):
        """从任务目标中提取关键词"""
        if not task_goal:
            return ""
        
        # 提取任务中的关键动词和名词
        import re
        keywords = []
        
        # 提取动作关键词
        action_words = re.findall(r'(设计|创作|描述|分析|规划|构建|完善)([^，。\n]{5,20})', task_goal)
        for action, target in action_words:
            keywords.append(f"{action}{target}")
        
        # 提取主题关键词
        theme_patterns = [r'(角色|人物|主角|配角)', r'(情节|剧情|故事|冲突)', r'(世界|背景|设定|环境)']
        for pattern in theme_patterns:
            matches = re.findall(pattern, task_goal)
            keywords.extend(matches)
        
        return " ".join(keywords[:6])
    








