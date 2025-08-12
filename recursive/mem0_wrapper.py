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






    def story_add(self, content, content_type, task_info):
        task_id = task_info.get("id")
        if not task_id:
            return
        task_type = task_info.get("task_type")
        task_goal = task_info.get("goal")
        dependency = task_info.get("dependency")
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        task_str = json.dumps(task_info, ensure_ascii=False)


        if content_type == "story_content":
            category = "story"
        elif content_type in ["task_update", "task_decomposition", "design_result"]:
            category = "design"


        mem0_content = ""
        if content_type == "story_content":
            mem0_content = content
        elif content_type == "task_decomposition":
            mem0_content = f"""任务：{task_str}\n规划分解结果：{content}"""
        elif content_type == "design_result":
            mem0_content = f"""任务：{task_str}\n设计结果：{content}"""
        elif content_type == "task_update":
            mem0_content = f"""任务：{task_str}\n更新任务目标：{content}"""

        
        parent_task_id = ".".join(task_id.split(".")[:-1]) if task_id and "." in task_id else ""
        mem_metadata = {
            "category": category,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "hierarchy_level": len(task_id.split(".")),
            "parent_task_id": parent_task_id,
            "dependency": dependency_str,
            "dependency_count": len(dependency),
            "content_length": len(mem0_content),
            "content_hash": hash(mem0_content) % 10000
        }
        
        
        self.client.add(
            mem0_content,
            user_id=f"{self.user_id_pre}_{category}",
            metadata=mem_metadata
        )
        
        logger.info(f"mem0 story_add() content_length={len(mem0_content)}, task_id={task_id}, metadata={mem_metadata}")





        

    def get_story_outer_graph_dependent(self, task_info, same_graph_dependent_designs, latest_content):
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        cur_hierarchy_level = len(task_id.split(".")) if task_id else 1
        
        # 构建多层次查询策略
        query_strategies = []
        
        # 策略1: 基于任务目标的关键词提取
        goal_keywords = self._extract_keywords_from_goal(task_goal)
        if goal_keywords:
            query_strategies.append(" ".join(goal_keywords))
        
        # 策略2: 基于任务类型的相关设计
        if task_type == "write":
            query_strategies.extend([
                "结构划分 字数分配 章节设计",
                "角色设计 情节设计 世界观设计",
                "开篇设计 悬念设计 节奏设计"
            ])
        elif task_type == "think":
            query_strategies.extend([
                "设计方案 具体方案 执行指导",
                "框架设计 要素分析 操作指南"
            ])
        
        # 策略3: 基于层级关系的父任务设计
        if "." in task_id:
            parent_id = ".".join(task_id.split(".")[:-1])
            query_strategies.append(f"任务{parent_id}")
        
        # 策略4: 包含同图依赖设计
        if same_graph_dependent_designs:
            query_strategies.append(same_graph_dependent_designs[:200])  # 限制长度
        
        # 执行多策略检索并合并结果
        all_contents = []
        seen = set()
        
        for query in query_strategies:
            if not query.strip():
                continue
                
            results = self.client.search(
                query=query,
                user_id=f"{self.user_id_pre}_design",
                limit=20,  # 每个策略限制结果数
                filters={
                    "content_type": "design_result",
                    "hierarchy_level": {"gte": 1, "lte": cur_hierarchy_level - 1}
                }
            )
            
            results = results if isinstance(results, list) else results.get('results', [])
            
            for r in results:
                content = r.get('memory', '')
                content_hash = hash(content)
                if content_hash not in seen and len(content) > 20:  # 过滤过短内容
                    all_contents.append({
                        'content': content,
                        'score': r.get('score', 0),
                        'metadata': r.get('metadata', {})
                    })
                    seen.add(content_hash)
        
        # 按相关性分数排序并选择最相关的内容
        all_contents.sort(key=lambda x: x['score'], reverse=True)
        final_contents = [item['content'] for item in all_contents[:15]]  # 最多15个最相关结果
        
        final_content = "\n\n".join(final_contents)
        logger.info(f"mem0 get_story_outer_graph_dependent() strategies={len(query_strategies)}, results={len(final_contents)}, cur_hierarchy_level={cur_hierarchy_level}")
        return final_content
    
    def _extract_keywords_from_goal(self, goal):
        """从任务目标中提取关键词"""
        if not goal:
            return []
        
        # 提取中文关键词的简单规则
        import re
        keywords = []
        
        # 提取专有名词和重要概念
        patterns = [
            r'第[一二三四五六七八九十\d]+[章幕卷]',  # 章节信息
            r'[\u4e00-\u9fff]{2,6}(?=设计|规划|分析|框架)',  # 设计相关
            r'(?:角色|人物|主角|反派|配角)[\u4e00-\u9fff]*',  # 角色相关
            r'(?:情节|剧情|故事|叙事)[\u4e00-\u9fff]*',  # 情节相关
            r'(?:世界观|背景|设定)[\u4e00-\u9fff]*',  # 世界观相关
            r'(?:开篇|结尾|高潮|转折)[\u4e00-\u9fff]*',  # 结构相关
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, goal)
            keywords.extend(matches)
        
        # 提取重要的双字词
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', goal)
        important_words = [w for w in words if any(key in w for key in 
                          ['设计', '规划', '分析', '框架', '结构', '角色', '情节', '世界', '开篇', '悬念'])]
        keywords.extend(important_words)
        
        return list(set(keywords))[:10]  # 去重并限制数量
        


    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        task_goal = task_info.get('goal', '')
        background_contents = self.get_story_content_background(task_goal, same_graph_dependent_designs, latest_content)
        plot_contents = self.get_story_content_plot(task_goal, same_graph_dependent_designs, latest_content)
        final_contents = []
        if background_contents:
            final_contents.append("=== 故事背景信息 ===")
            final_contents.extend(background_contents)
        if plot_contents:
            final_contents.append("=== 情节发展信息 ===")
            final_contents.extend(plot_contents)
        return "\n".join(final_contents)

    def get_story_content_background(self, task_goal, same_graph_dependent_designs, latest_content):
        background_tags = [
            "世界背景:", "规则体系:", "文化背景:", "科技魔法:", "社会结构:",
            "角色档案:", "角色关系:", "场景设置:"
        ]
        
        query_parts = []
        query_parts.append(f"({' OR '.join(background_tags)})")
        query_parts.append(task_goal)
        if same_graph_dependent_designs:
            query_parts.append(same_graph_dependent_designs)
        
        query = " ".join(query_parts)
        
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_story",
            limit=500
        )
        results = results if isinstance(results, list) else results.get('results', [])
        contents = []
        seen = set()
        for r in results:
            content = r.get('memory', '')
            content_hash = hash(content)
            if content_hash not in seen:
                contents.append(content)
                seen.add(content_hash)
        logger.info(f"mem0 get_story_content_background() query=\n{query}\n, contents=\n{contents}")
        return contents

    def get_story_content_plot(self, task_goal, same_graph_dependent_designs, latest_content):
        plot_tags = [
            "关键事件:", "情节转折:", "冲突设置:", "因果关系:", "情节线索:",
            "时间节点:", "角色状态:", "角色成长:", "角色动机:",
            "关键对话:", "信息透露:", "情感表达:", "观点冲突:",
            "悬念设置:", "伏笔线索:", "未解谜题:", "预示暗示:"
        ]
        
        query_parts = []
        query_parts.append(f"({' OR '.join(plot_tags)})")
        query_parts.append(task_goal)
        if same_graph_dependent_designs:
            query_parts.append(same_graph_dependent_designs)
        
        # 添加最新内容上下文以保持连续性
        length = 2000
        if latest_content:
            latest_context = latest_content[-length:] if len(latest_content) > length else latest_content
            # 提取人物名称
            import re
            names = re.findall(r'[\u4e00-\u9fff]{2,4}(?=说|道|想|看|听|感到|发现)', latest_context)
            if names:
                query_parts.append(" ".join(set(names[:10])))  # 最多10个人物名
        
        query = " ".join(query_parts)
        
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_story",
            limit=500
        )
        results = results if isinstance(results, list) else results.get('results', [])
        
        # 按时间排序情节信息
        results.sort(key=lambda x: x.get('metadata', {}).get('timestamp', ''))
        
        contents = []
        seen = set()
        for r in results:
            content = r.get('memory', '')
            content_hash = hash(content)
            if content_hash not in seen:
                contents.append(content)
                seen.add(content_hash)
        logger.info(f"mem0 get_story_content_plot() query=\n{query}\n, contents=\n{contents}")
        return contents


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
    
    









