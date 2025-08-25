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
分析、评估、审查 项目 的 RAG 的质量和效果，项目的目标是创作出爆款的超长篇网络小说，从项目角度，给出全面的报告，并指出其最大的优势和可以进一步强化的方向。


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

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        cache_dir = os.path.join(project_root, ".cache", "mem0_text")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_text = diskcache.Cache(cache_dir, size_limit=1024 * 1024 * 300)

        cache_dir = os.path.join(project_root, ".cache", "mem0_design")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_design = diskcache.Cache(cache_dir, size_limit=1024 * 1024 * 300)
        
        cache_dir = os.path.join(project_root, ".cache", "mem0_full_plan")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_full_plan = diskcache.Cache(cache_dir, size_limit=1024 * 1024 * 100)
        
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
        if not task_id or task_id == "" or task_id == "0":
            raise ValueError("Task ID not found in task_info {task_id} \n task_info: {task_info}")
        task_type = task_info.get("task_type")
        task_goal = task_info.get("goal")
        dependency = task_info.get("dependency", [])
        task_str = json.dumps(task_info, ensure_ascii=False)
        dependency_str = json.dumps(dependency, ensure_ascii=False)
        logger.info(f"mem0 add() {task_info}")

        if content_type == "text_content":
            category = "text"
        elif content_type in ["task_update", "task_decomposition", "design_result"]:
            category = "design"

        mem0_content = ""
        if content_type == "text_content":
            mem0_content = content
            self.cache_text.clear()
        elif content_type == "task_decomposition":
            mem0_content = f"""任务：\n{task_str}\n规划分解结果：\n{content}"""
            self.cache_full_plan.clear()
        elif content_type == "design_result":
            mem0_content = f"""任务：\n{task_str}\n设计结果：\n{content}"""
            self.cache_design.clear()
        elif content_type == "task_update":
            mem0_content = f"""任务更新：\n{content}"""
            self.cache_full_plan.clear()

        parent_task_id = ".".join(task_id.split(".")[:-1]) if task_id and "." in task_id else ""
        mem_metadata = {
            "category": category,
            "content_type": content_type,
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "hierarchy_level": len(task_id.split(".")),
            "parent_task_id": parent_task_id,
            "dependency": dependency_str,
            # "content_length": len(mem0_content),
            # "content_hash": hashlib.md5(mem0_content.encode('utf-8')).hexdigest()
        }
        
        logger.info(f"mem0 add() {self.writing_mode}_{self.language}_{hashkey}_{category}\n{mem0_content}\n{mem_metadata}")
        self.get_client().add(
            mem0_content,
            user_id=f"{self.writing_mode}_{self.language}_{hashkey}_{category}",
            metadata=mem_metadata
        )

    def search(self, hashkey, querys, category, limit=1, filters=None):
        if not querys:
            return []
        
        unique_queries = list(dict.fromkeys([q.strip() for q in querys if q and q.strip()]))
        if not unique_queries:
            return []
        
        user_id = f"{self.writing_mode}_{self.language}_{hashkey}_{category}"
        
        if len(unique_queries) == 1:
            query = unique_queries[0]
        elif len(unique_queries) <= 8:
            query = " OR ".join([f"({q})" for q in unique_queries])
        elif len(unique_queries) <= 15:
            primary_queries = unique_queries[:8]
            secondary_queries = unique_queries[8:15]
            primary_part = " OR ".join([f"({q})" for q in primary_queries])
            secondary_part = " ".join(secondary_queries)
            query = f"({primary_part}) {secondary_part}"
            logger.info(f"Using hybrid strategy: {len(primary_queries)} OR queries + {len(secondary_queries)} AND queries")
        else:
            core_queries = unique_queries[:6]
            important_queries = unique_queries[6:12]
            supplement_queries = unique_queries[12:20]
            
            core_part = " OR ".join([f"({q})" for q in core_queries])
            important_part = " ".join(important_queries) if important_queries else ""
            supplement_part = " ".join(supplement_queries) if supplement_queries else ""
            
            query_parts = [f"({core_part})"]
            if important_part:
                query_parts.append(important_part)
            if supplement_part:
                query_parts.append(supplement_part)
            
            query = " ".join(query_parts)
            logger.info(f"Using layered strategy: {len(core_queries)} core + {len(important_queries)} important + {len(supplement_queries)} supplement queries")

            if len(unique_queries) > 20:
                logger.info(f"Truncated {len(unique_queries) - 20} queries to maintain performance")
        
        logger.info(f"mem0 search() {user_id}\nOptimized query: {query[:200]}...\nTotal keywords: {len(unique_queries)}\nLimit: {limit}\nFilters: {filters}")
        
        try:
            results = self.get_client().search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters
            )
            
            if isinstance(results, dict):
                final_results = results.get('results', [])
            elif isinstance(results, list):
                final_results = results
            else:
                final_results = []
            
            final_results = self._apply_hybrid_ranking(final_results, unique_queries, category)
            
            logger.info(f"mem0 search() completed: {len(final_results)} results for {len(unique_queries)} queries")
            return final_results
            
        except Exception as e:
            logger.error(f"Optimized search failed: {e}")
            logger.info("Falling back to batch retrieval strategy")
            
            try:
                all_fallback_results = []
                batch_size = 8
                for i in range(0, min(len(unique_queries), 16), batch_size):
                    batch_queries = unique_queries[i:i + batch_size]
                    batch_query = " OR ".join([f"({q})" for q in batch_queries])
                    
                    batch_results = self.get_client().search(
                        query=batch_query,
                        user_id=user_id,
                        limit=max(limit // 2, 50), 
                        filters=filters
                    )
                    
                    if isinstance(batch_results, dict):
                        batch_final = batch_results.get('results', [])
                    elif isinstance(batch_results, list):
                        batch_final = batch_results
                    else:
                        batch_final = []
                    
                    all_fallback_results.extend(batch_final)
                
                seen_memories = set()
                unique_results = []
                for result in all_fallback_results:
                    memory = result.get('memory', '')
                    if memory and memory not in seen_memories:
                        unique_results.append(result)
                        seen_memories.add(memory)
                
                unique_results = self._apply_hybrid_ranking(unique_results, unique_queries, category)
                final_fallback_results = unique_results[:limit]
                
                logger.info(f"Batch fallback search completed: {len(final_fallback_results)} results from {len(unique_queries)} queries")
                return final_fallback_results
                
            except Exception as e2:
                logger.error(f"Batch fallback search also failed: {e2}")
                try:
                    simple_query = " ".join(unique_queries[:5])
                    logger.info(f"Final fallback to simple query: {simple_query}")
                    
                    simple_results = self.get_client().search(
                        query=simple_query,
                        user_id=user_id,
                        limit=limit,
                        filters=filters
                    )
                    
                    if isinstance(simple_results, dict):
                        return simple_results.get('results', [])
                    elif isinstance(simple_results, list):
                        return simple_results
                    else:
                        return []
                        
                except Exception as e3:
                    logger.error(f"All search strategies failed: {e3}")
                    return []
    
    def _apply_hybrid_ranking(self, results, query_terms, category):
        if not results or not query_terms:
            return results
        
        try:
            enhanced_results = []
            for result in results:
                memory = result.get('memory', '').lower()
                original_score = result.get('score', 0.0)
                
                # 计算关键词匹配分数
                keyword_match_score = 0
                for term in query_terms:
                    if term.lower() in memory:
                        # 精确匹配加分更多
                        keyword_match_score += 2.0
                    else:
                        # 部分匹配检查
                        term_chars = set(term.lower())
                        memory_chars = set(memory)
                        overlap = len(term_chars & memory_chars) / len(term_chars) if term_chars else 0
                        if overlap > 0.5:  # 50%以上字符重叠
                            keyword_match_score += overlap
                
                # 计算内容质量分数
                content_quality_score = 0
                metadata = result.get('metadata', {})
                content_length = len(memory)
                
                # 内容长度评分
                if 50 <= content_length <= 1000:
                    content_quality_score += 1.0
                elif content_length > 1000:
                    content_quality_score += 0.8
                else:
                    content_quality_score += 0.5
                
                # 時间新鲜度评分
                if 'timestamp' in metadata:
                    try:
                        timestamp = datetime.fromisoformat(metadata['timestamp'])
                        time_diff = (datetime.now() - timestamp).total_seconds()
                        if time_diff <= 3600:  # 1小时内
                            content_quality_score += 1.5
                        elif time_diff <= 86400:  # 1天内
                            content_quality_score += 1.0
                        elif time_diff <= 604800:  # 1周内
                            content_quality_score += 0.5
                    except:
                        pass
                
                # 类别特定评分
                category_score = 0
                if category == "design":
                    # 设计类内容优先考虑层级和结构化程度
                    if 'hierarchy_level' in metadata:
                        category_score += metadata.get('hierarchy_level', 0) * 0.1
                    if any(marker in memory for marker in ['###', '|', '```', '-']):
                        category_score += 0.5  # 结构化内容加分
                elif category == "text":
                    # 文本类内容优先考虑连贯性和完整性
                    if any(marker in memory for marker in ['。', '！', '？', '\n']):
                        category_score += 0.3  # 完整句子加分
                
                # 综合评分计算
                final_score = (
                    original_score * 0.4 +  # 原始语义相似度权重
                    keyword_match_score * 0.35 +  # 关键词匹配权重
                    content_quality_score * 0.15 +  # 内容质量权重
                    category_score * 0.1  # 类别特定权重
                )
                
                enhanced_result = result.copy()
                enhanced_result['hybrid_score'] = final_score
                enhanced_result['keyword_match_score'] = keyword_match_score
                enhanced_result['content_quality_score'] = content_quality_score
                enhanced_results.append(enhanced_result)
            
            enhanced_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            
            logger.info(f"Applied hybrid ranking to {len(results)} results, top score: {enhanced_results[0].get('hybrid_score', 0):.3f}" if enhanced_results else "No results to rank")
            return enhanced_results
            
        except Exception as e:
            logger.warning(f"Hybrid ranking failed: {e}, falling back to original order")
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
        
        # 优化的多维度排序策略，提升检索结果质量
        def sort_key_design(x):
            meta = x.get('metadata', {})
            
            # 解析时间戳
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
            except (ValueError, TypeError):
                timestamp = datetime.min
            
            # 多维度评分系统
            hierarchy_level = meta.get('hierarchy_level', 0)
            base_score = x.get('score', 0.0)
            content_length = meta.get('content_length', 0)
            
            # 层级权重：越高层级越重要
            hierarchy_weight = hierarchy_level * 100
            
            # 相关性权重：基础分数 * 50
            relevance_weight = base_score * 50
            
            # 内容丰富度权重：适中长度最优
            if content_length < 50:
                length_weight = content_length * 0.1  # 过短内容降权
            elif content_length <= 500:
                length_weight = content_length * 0.3  # 适中内容加权
            else:
                length_weight = 500 * 0.3 + (content_length - 500) * 0.1  # 过长内容稍微降权
            
            # 时间新鲜度权重：较新的内容加分
            time_diff_days = (datetime.now() - timestamp).days
            if time_diff_days <= 1:
                time_weight = 20  # 最新内容
            elif time_diff_days <= 7:
                time_weight = 15  # 近期内容
            elif time_diff_days <= 30:
                time_weight = 10  # 中期内容
            else:
                time_weight = 5   # 早期内容
            
            # 综合评分
            final_score = hierarchy_weight + relevance_weight + length_weight + time_weight
            
            return final_score
        
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
        
        # 优化的文本内容排序策略，优先时间新鲜度和相关性
        def sort_key_text(x):
            meta = x.get('metadata', {})
            
            # 解析时间戳
            ts_str = meta.get('timestamp', '')
            try:
                timestamp = datetime.fromisoformat(ts_str) if ts_str else datetime.min
            except (ValueError, TypeError):
                timestamp = datetime.min
            
            # 多维度评分系统
            base_score = x.get('score', 0.0)
            hierarchy_level = meta.get('hierarchy_level', 0)
            content_length = meta.get('content_length', 0)
            
            # 时间新鲜度权重：最重要的因素
            time_diff_seconds = (datetime.now() - timestamp).total_seconds()
            if time_diff_seconds <= 3600:  # 1小时内
                time_weight = 200
            elif time_diff_seconds <= 86400:  # 1天内
                time_weight = 150
            elif time_diff_seconds <= 604800:  # 1周内
                time_weight = 100
            elif time_diff_seconds <= 2592000:  # 1个月内
                time_weight = 50
            else:
                time_weight = 10
            
            # 相关性权重：高相关性内容优先
            relevance_weight = base_score * 80
            
            # 层级权重：与当前任务层级越接近越好
            task_hierarchy = len(task_id.split(".")) if task_id else 1
            level_diff = abs(hierarchy_level - task_hierarchy)
            if level_diff == 0:
                hierarchy_weight = 30  # 相同层级
            elif level_diff == 1:
                hierarchy_weight = 20  # 相邻层级
            elif level_diff == 2:
                hierarchy_weight = 10  # 较远层级
            else:
                hierarchy_weight = 5   # 很远层级
            
            # 内容丰富度权重：中等长度内容更有价值
            if content_length < 100:
                length_weight = content_length * 0.1
            elif content_length <= 1000:
                length_weight = 10 + (content_length - 100) * 0.02
            else:
                length_weight = 10 + 900 * 0.02 + (content_length - 1000) * 0.005
            
            # 综合评分
            final_score = time_weight + relevance_weight + hierarchy_weight + length_weight
            
            return final_score
        
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
        task_goal = task_info.get('goal', '')
        
        cache_key = f"full_plan_{self.writing_mode}_{self.language}_{hashkey}_{task_id}"
        cached_result = self.cache_full_plan.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # 尝试使用Cypher查询进行高效路径检索
        cypher_result = self._get_full_plan_via_cypher(hashkey, task_id)
        if cypher_result:
            self.cache_full_plan.set(cache_key, cypher_result)
            logger.info(f"mem0 get_full_plan_cypher() {self.writing_mode}_{self.language}_{hashkey}\n{task_goal}\n{cypher_result}")
            return cypher_result
        
        # 降级到原有的批量查询策略
        logger.info("Cypher query failed, falling back to batch search strategy")
        return self._get_full_plan_fallback(hashkey, task_info)
    
    def _get_full_plan_via_cypher(self, hashkey, task_id):
        """
        使用Cypher查询直接从Memgraph获取任务层级路径
        提升查询效率，减少网络往返
        """
        try:
            # 构建任务ID层级链
            to_task_ids = []
            current_id = task_id
            to_task_ids.append(current_id)
            while "." in current_id:
                current_id = ".".join(current_id.split(".")[:-1])
                to_task_ids.append(current_id)
            to_task_ids = sorted(to_task_ids, key=lambda x: len(x.split(".")))
            
            # 通过mem0的图数据库连接执行Cypher查询
            graph_store = self.get_client().graph_store
            if not hasattr(graph_store, 'graph'):
                return None
                
            # 构建Cypher查询语句 - 查找包含目标任务ID的所有节点
            task_goals = []
            user_id = f"{self.writing_mode}_{self.language}_{hashkey}"
            
            for pid in to_task_ids:
                # 使用Cypher查询查找包含特定任务ID的实体
                cypher_query = """
                MATCH (e:Entity)
                WHERE e.user_id = $user_id 
                  AND (e.name CONTAINS $task_id OR e.text CONTAINS $task_id)
                RETURN e.name, e.text
                LIMIT 5
                """
                
                try:
                    result = graph_store.graph.run(cypher_query, 
                                                  user_id=user_id, 
                                                  task_id=pid)
                    
                    for record in result:
                        name = record.get('e.name', '')
                        text = record.get('e.text', '')
                        content = text if text else name
                        
                        if content and pid in content:
                            task_goals.append(f"[{pid}]: {content}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Cypher query failed for task {pid}: {e}")
                    continue
            
            if task_goals:
                # 按任务ID层级排序
                def sort_key(item):
                    task_id_match = item.split("]: ")[0].replace("[", "")
                    return len(task_id_match.split("."))
                
                task_goals.sort(key=sort_key)
                return "\n".join(task_goals)
                
        except Exception as e:
            logger.warning(f"Cypher-based full plan query failed: {e}")
            return None
        
        return None
    
    def _get_full_plan_fallback(self, hashkey, task_info):
        """
        原有的批量查询策略作为降级方案
        """
        task_id = task_info.get("id", "")
        task_goal = task_info.get('goal', '')
        
        # 构建任务ID层级链
        to_task_ids = []
        current_id = task_id
        to_task_ids.append(current_id)
        while "." in current_id:
            current_id = ".".join(current_id.split(".")[:-1])
            to_task_ids.append(current_id)
        to_task_ids = sorted(to_task_ids, key=lambda x: len(x.split(".")))
        
        # 构建全面的批量查询，使用多种查询模式提升匹配成功率
        task_id_patterns = []
        for pid in to_task_ids:
            # 基础查询模式
            task_id_patterns.extend([
                f"任务id为{pid}",
                f"id:{pid}",
                f"[{pid}]",
                f'"id":"{pid}"',
                f'"id": "{pid}"',
                f"任务{pid}",
                f"task_id:{pid}",
                f"任务编号{pid}"
            ])
        
        # 智能分批查询策略，避免查询过长同时保证覆盖度
        all_results = []
        batch_size = 12  # 增加批次大小
        
        for i in range(0, len(task_id_patterns), batch_size):
            batch_patterns = task_id_patterns[i:i + batch_size]
            batch_query = " OR ".join(batch_patterns)
            
            try:
                batch_results = self.get_client().search(
                    query=batch_query,
                    user_id=f"{self.writing_mode}_{self.language}_{hashkey}_design",
                    limit=100  # 每批次增加限制数量
                )
                
                batch_results = batch_results if isinstance(batch_results, list) else batch_results.get('results', [])
                all_results.extend(batch_results)
                
            except Exception as e:
                logger.warning(f"Batch query failed for patterns {i//batch_size + 1}: {e}")
                continue
        
        # 合并所有批次结果，去重
        seen_memories = set()
        unique_results = []
        for result in all_results:
            memory = result.get('memory', '')
            if memory and memory not in seen_memories:
                unique_results.append(result)
                seen_memories.add(memory)
        
        # 解析和匹配任务信息，增强匹配策略
        task_goals = []
        found_task_ids = set()
        
        # 优先使用批量查询结果，采用多种匹配模式
        for result in unique_results:
            memory_content = result.get('memory', '')
            if not memory_content:
                continue
                
            # 逐个尝试匹配所有任务ID
            for pid in to_task_ids:
                if pid in found_task_ids:
                    continue
                    
                # 多种匹配模式，提高匹配成功率
                memory_lower = memory_content.lower()
                match_patterns = [
                    f"任务id为{pid}",
                    f"id:{pid}",
                    f"[{pid}]",
                    f'"id":"{pid}"',
                    f'"id": "{pid}"',
                    f"任务{pid}",
                    f"task_id:{pid}",
                    f"任务编号{pid}",
                    f"id={pid}",
                    f"taskid:{pid}",
                    f"编号{pid}"
                ]
                
                # 检查是否匹配任何一种模式
                if any(pattern.lower() in memory_lower for pattern in match_patterns):
                    task_goals.append(f"[{pid}]: {memory_content}")
                    found_task_ids.add(pid)
                    break
        
        # 对于批量查询未找到的任务，进行单独查询，使用更多查询词
        for pid in to_task_ids:
            if pid not in found_task_ids:
                # 使用多个查询词增加找到的概率
                individual_queries = [
                    f"任务id为{pid}的详细目标(goal)",
                    f"任务{pid} 目标 详细内容",
                    f"id:{pid} goal task",
                    f"[{pid}] 任务计划",
                    f"任务编号{pid}对应的内容"
                ]
                
                for query in individual_queries:
                    try:
                        individual_results = self.get_client().search(
                            query=query,
                            user_id=f"{self.writing_mode}_{self.language}_{hashkey}_design",
                            limit=3  # 每个查询返回多个结果
                        )
                        individual_results = individual_results if isinstance(individual_results, list) else individual_results.get('results', [])
                        
                        # 检查结果中是否包含目标任务ID
                        for result in individual_results:
                            memory = result.get('memory', '')
                            if memory and pid in memory:
                                task_goals.append(f"[{pid}]: {memory}")
                                found_task_ids.add(pid)
                                break
                        
                        if pid in found_task_ids:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Individual query failed for task {pid} with query '{query}': {e}")
                        continue
        
        # 按任务ID层级排序
        def sort_key(item):
            task_id_match = item.split("]: ")[0].replace("[", "")
            return len(task_id_match.split("."))
        
        task_goals.sort(key=sort_key)
        result = "\n".join(task_goals)
        
        cache_key = f"full_plan_{self.writing_mode}_{self.language}_{hashkey}_{task_info.get('id', '')}"
        self.cache_full_plan.set(cache_key, result)
        
        logger.info(f"mem0 get_full_plan_fallback() {self.writing_mode}_{self.language}_{hashkey}\n{task_info.get('goal', '')}\n{task_goals}")
        return result


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
