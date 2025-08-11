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

        self.config = {
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
            "llm": {
                "provider": "openai",
                "config": {
                    "temperature": 0.0,
                    "max_tokens": 131072
                }
            },
            # "llm": {
            #     "provider": "litellm",
            #     "config": {
            #         "model": os.getenv("fast_model"),
            #         "temperature": 0.0,
            #         "max_tokens": 131072
            #     }
            # },
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

        temp = os.getenv("OPENROUTER_API_KEY")
        if self.config["llm"]["provider"] == "openai":
            # os.environ["OPENROUTER_API_KEY"] = ""
            if os.environ.get("OPENROUTER_API_KEY"):
                self.config["llm"]["config"] = {
                    "model": "deepseek/deepseek-chat-v3-0324:free",
                    "temperature": 0.0,
                    "max_tokens": 131072
                }
            else:
                self.config["llm"]["config"] = {
                    "model": "deepseek-ai/DeepSeek-V3",
                    "temperature": 0.0,
                    "max_tokens": 131072
                }

        self.client = Mem0Memory.from_config(config_dict=self.config)
        os.environ["OPENROUTER_API_KEY"] = temp
        
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
        task_id = task_info.get("id", "")
        task_type = task_info.get("task_type", "")
        task_goal = task_info.get("goal", "")
        dependency = task_info.get("dependency", [])
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        task_str = json.dumps(task_info, ensure_ascii=False)

        if content_type == "story_content":
            category = "story"
        elif content_type in ["task_update", "task_decomposition", "design_result"]:
            category = "design"
        else:
            return
        
        # 预处理内容，对于复杂的设计结果进行分块处理
        if content_type == "design_result" and len(content) > 3000:
            self._add_complex_design_content(content, task_info, category)
        else:
            # 标准处理流程
            mem0_content = self._format_content_for_storage(content, content_type, task_str)
            self._add_single_content(mem0_content, task_info, category, content_type, content)

    def _format_content_for_storage(self, content, content_type, task_str):
        """格式化内容用于存储"""
        if content_type == "story_content":
            return content
        elif content_type == "task_decomposition":
            return f"""任务：{task_str}\n规划分解结果：{content}"""
        elif content_type == "design_result":
            return f"""任务：{task_str}\n设计结果：{content}"""
        elif content_type == "task_update":
            return f"""任务：{task_str}\n更新任务目标：{content}"""
        return content

    def _add_complex_design_content(self, content, task_info, category):
        """处理复杂的设计内容，分块存储"""
        task_id = task_info.get("id", "")
        task_str = json.dumps(task_info, ensure_ascii=False)
        
        # 按主要结构分块
        chunks = self._split_design_content(content)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:  # 跳过过短的块
                continue
                
            # 为每个块创建描述性标题
            chunk_title = self._extract_chunk_title(chunk)
            formatted_content = f"""任务：{task_str}\n设计结果片段[{chunk_title}]：{chunk}"""
            
            # 添加块索引到元数据
            chunk_metadata = {
                "chunk_index": i,
                "chunk_title": chunk_title,
                "total_chunks": len(chunks),
                "is_chunk": True
            }
            
            self._add_single_content(formatted_content, task_info, category, "design_result", chunk, chunk_metadata)

    def _split_design_content(self, content):
        """将复杂设计内容分割成有意义的块"""
        chunks = []
        
        # 按markdown标题分割
        import re
        sections = re.split(r'\n(?=#{1,4}\s)', content)
        
        if len(sections) > 1:
            # 有明确的标题结构
            for section in sections:
                if section.strip():
                    chunks.append(section.strip())
        else:
            # 按表格分割
            table_pattern = r'\|[^|]+\|[^|]+\|'
            if re.search(table_pattern, content):
                parts = re.split(r'\n(?=\|)', content)
                current_chunk = ""
                for part in parts:
                    if len(current_chunk + part) > 1500:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part
                    else:
                        current_chunk += "\n" + part
                if current_chunk:
                    chunks.append(current_chunk.strip())
            else:
                # 按段落分割
                paragraphs = content.split('\n\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk + para) > 1500:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para
                    else:
                        current_chunk += "\n\n" + para
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _extract_chunk_title(self, chunk):
        """从内容块中提取描述性标题"""
        import re
        
        # 尝试提取markdown标题
        title_match = re.search(r'^#{1,4}\s*(.+)', chunk, re.MULTILINE)
        if title_match:
            return title_match.group(1).strip()
        
        # 尝试提取表格主题
        if '|' in chunk:
            lines = chunk.split('\n')
            for line in lines:
                if '|' in line and not line.strip().startswith('|---'):
                    return "表格数据"
        
        # 提取关键词
        keywords = re.findall(r'[\u4e00-\u9fff]{2,6}(?=设计|规划|分析|框架|方案)', chunk)
        if keywords:
            return keywords[0] + "设计"
        
        # 默认标题
        return "设计内容"

    def _add_single_content(self, mem0_content, task_info, category, content_type, original_content, extra_metadata=None):
        """添加单个内容到mem0"""
        task_id = task_info.get("id", "")
        dependency = task_info.get("dependency", [])
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        
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
            "content_length": len(original_content),
            "content_hash": hash(original_content) % 10000
        }
        
        # 添加额外元数据
        if extra_metadata:
            mem_metadata.update(extra_metadata)
        
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
    
    









