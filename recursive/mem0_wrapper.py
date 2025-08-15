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
# import diskcache as dc
from recursive.agent.prompts.story_zh.mem import (
    MEM_STORY_FACT,
    MEM_STORY_UPDATE
)


class Mem0:
    def __init__(self, root_node, config):
        # 确定写作模式（story, book, report）
        self.writing_mode = config.get("writing_mode", "story")
        self.root_node = root_node
        
        # # 使用diskcache作为查询缓存
        # cache_dir = f"./.mem0/cache_{self.root_node.hashkey}"
        # os.makedirs(cache_dir, exist_ok=True)
        # self._query_cache = dc.Cache(cache_dir, size_limit=100*1024*1024)  # 100MB缓存限制
        

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
        
        # cache_key = f"{self.user_id_pre}_{category}_{query}_{limit}_{str(filters)}"
        # if cache_key in self._query_cache:
        #     return self._query_cache[cache_key]
        
        logger.info(f"mem0 search() {category} query=\n{query}")
        results = self.client.search(
            query=query,
            user_id=f"{self.user_id_pre}_{category}",
            limit=limit,
            filters=filters
        )
        logger.info(f"mem0 search() results=\n{results}")
        # self._query_cache[cache_key] = results
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
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 to_run_outer_graph_dependent
        检索故事创作中的 设计结果 ，为故事创作提供关键的设计上下文支持
        保障设计一致性 ：通过检索相关设计结果，确保新创作内容与既有设计保持一致

        参数:
        - task_info: 任务信息字典，包含goal、id、task_type等
        - same_graph_dependent_designs: 同层的设计方案
        - latest_content: 最新内容文本

        检索词:
        参考 agent/prompts/story_zh 中的 mem.py 中的 MEM_STORY_FACT 中的标签体系
        
        核心设定类：世界观名称、核心特征、世界规则、力量体系节点、力量等级划分、核心冲突根源、主角核心目标、关键地域设定、势力分布框架、资源体系规则、设定弹性空间（可扩展边界/留白区域）、跨体系交互规则（力量碰撞/资源兑换）、时间线锚点（历史节点/时间流速差异）
        情节主线类：主线任务阶段、阶段性目标、关键转折点、伏笔回收节点、反派势力层级、主线分支关联、重要事件排序、卷/篇划分逻辑、结局走向框架、危机升级路径、支线任务权重（关联度/回收阈值）、节奏锚点标记（低谷阈值/高潮铺垫章节数）、跨卷线索延续性（悬念衔接/风格统一度）
        角色设定类：主角核心特质、主角特殊体质/天赋、重要配角功能定位、角色关系网框架、角色成长弧线、反派核心动机、关键角色身份设定、阵营划分原则、角色核心执念、角色标签系统（标志性言行/禁忌领域）、关系网动态阈值（质变节点/忠诚度波动范围）、角色功能冗余度（不可替代性/差异化设计）
        爽点框架类：爽点类型分布（打脸/升级/寻宝/艳遇）、爽点强度梯度（小爽/中爽/大高潮）、爽点间隔周期、反套路节点、核心爽点重复阈值、跨卷爽点呼应设计、爽点创新维度（变异设计/跨类型融合）、读者反馈响应机制（高频爽点强化/争议调整阈值）、爽点与主线绑定度（服务主线/独立篇幅限制）
        世界扩展类：地图解锁条件、新区域核心规则、跨区域势力关联、世界真相揭露阶段、新种族/文明引入契机、区域专属资源类型、地域风险等级、区域文化符号（习俗禁忌/建筑服饰特色）、探索进度条（隐藏探索度/强制线索）、世界风险梯度（危险量化指标/旅行成本收益比）
        角色长线类：配角成长关键节点、角色阶段性功能切换、反派迭代逻辑、角色退场/回归触发条件、阵营忠诚度变化阈值、核心角色黑化/洗白契机、角色“保鲜期”（功能有效期/登场热度阈值）、反派“升级逻辑”（实力跳跃区间/动机层次感）、角色“彩蛋回收”（路人反转空间/执念阶段性变体）
        """
        task_goal = task_info.get('goal', '')
        task_type = task_info.get('task_type', '')
        task_id = task_info.get('id', '')
        
        all_queries = []

        # 添加任务目标作为核心检索词
        all_queries.append(task_goal)

        # 1. 核心框架类
        core_framework_keywords = [
            "故事总主线", "阶段划分", "核心目标", "关键转折点",
            "叙事架构", "结构设计", "整体框架"
        ]
        all_queries.extend(core_framework_keywords)
        
        # 2. 设定体系类 - 聚焦世界和角色的核心设定
        setting_system_keywords = [
            "世界观核心规则", "金手指设定", "核心道具设定", 
            "主要势力关系", "伏笔总览", "角色背景设定"
        ]
        all_queries.extend(setting_system_keywords)
        
        # 3. 爽点规划类 - 针对故事的情感设计
        pleasure_planning_keywords = [
            "核心爽点类型", "打脸对象层级", "升级节奏",
            "情感高潮", "冲突设计"
        ]
        all_queries.extend(pleasure_planning_keywords)

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
                all_queries.append(f"第{num}{level_name} 设计")
                # 前后层级上下文（用于获取连续性）
                try:
                    num_int = int(num)
                    if num_int > 1:
                        all_queries.append(f"第{num_int-1}{level_name} 设计")
                    all_queries.append(f"第{num_int+1}{level_name} 设计")
                except:
                    pass
        
        # 提取任务ID相关上下文 - 增加设计标识提高精准度
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id} 设计方案")
            # 添加当前任务ID
            all_queries.append(f"任务 {task_id} 设计方案")
        
        # 从最新内容提取关键词 - 聚焦与设计相关的元素
        if latest_content and isinstance(latest_content, str):
            content_keywords = self._extract_keywords_from_text(latest_content)
            # 只添加与设计相关的关键词
            design_related_keywords = [kw for kw in content_keywords if kw in ['设定', '架构', '框架', '规划', '设计']]
            all_queries.extend(design_related_keywords)

        # 从设计方案提取关键词 - 增强设计文档的相关性
        design_keywords = self._extract_keywords_from_markdown(same_graph_dependent_designs)
        all_queries.extend([f"{kw} 设计" for kw in design_keywords])
        
        # 合并所有关键词并去重
        final_queries = list(dict.fromkeys(all_queries))

        # 执行检索
        all_results = self.search(final_queries, "design", limit=500)
        
        # 处理结果：按相关性排序并过滤
        combined_results = []
        seen_contents = set()
        for result in all_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                # 过滤掉太短的内容
                if len(memory_content) > 100:
                    combined_results.append(memory_content)
                    seen_contents.add(memory_content)
                # 限制结果数量，避免信息过载
                if len(combined_results) >= 1500:
                    break
        return "\n\n".join(combined_results)











    def get_story_content(self, task_info, same_graph_dependent_designs, latest_content):
        """
        替换 agent/agents/regular.py 中的 get_llm_output 中的 memory.article
        检索已创作的故事正文内容，为后续创作提供上下文支持。
        优先考虑最新内容，确保创作的时效性和连续性

        参数:
        - task_info: 任务信息字典，包含goal、id、task_type等
        - same_graph_dependent_designs: 同层的设计方案
        - latest_content: 最新内容文本

        检索词:
        参考 agent/prompts/story_zh 中的 mem.py 中的 MEM_STORY_FACT 中的标签体系

        具体情节类：关键事件细节、近期剧情节点、爽点情节记录、打脸场景要素、战斗结果记录、任务完成状态、情节分支进展、角色生死状态、重大决策过程、情节“辐射范围”（影响数量/长期后遗症）、读者“记忆峰值”（高频讨论名场面/二刷情节标记）、情节“修正记录”（设定冲突修正/逻辑漏洞补坑）
        角色互动类：角色对话关键信息、角色行为习惯、临时阵营关系、近期情感互动、角色冲突点、承诺与恩怨记录、合作/敌对状态、角色信任度变化、对话伏笔内容、情感“量化标记”（信任度数值/名台词复用场景）、互动“潜台词”（弦外之音/未说出口的承诺）、冲突“遗留问题”（核心分歧/妥协条件有效期）
        场景与道具类：关键场景细节、道具状态（破损/升级）、技能掌握情况（熟练度/限制）、未回收细节、环境特征、物品获取记录、空间/时间节点标记、场景专属规则、道具“背景故事”（前主人经历/情绪价值）、场景“复用设计”（功能变化/标志性元素）、技能“副作用”（代价记录/进化线索）
        爽点细节类：近期爽点读者反馈、重复爽点规避记录、爽点留白位置、打脸方式创新点、升级带来的具体特权、读者高频讨论的爽点场景、爽点“感官描写模板”（套路/差异化标记）、反套路“成功案例”（创新结构/读者认可类型）、爽点“情绪峰值”（评论热词/单章字数占比）
        伏笔关联类：伏笔埋设章节+具体描述、伏笔关联角色/道具、伏笔回收倒计时、伏笔线索扩散节点、伪伏笔排除标记、跨卷伏笔衔接细节、伏笔“紧急度”（必须回收/可放弃分类/提醒节点）、伏笔“线索扩散”（信息掌握者/误读线索角色）、回收“成本”（资源门槛/连锁反应）
        节奏调控类：密集爽点后的缓冲情节、低谷期长度（章节数）、卷末高潮与下卷衔接点、节奏拖沓预警标记、剧情加速触发事件、日常情节占比、节奏“读者耐受度”（低谷流失率/爽点疲劳阈值）、章节“钩子有效性”（类型记录/上钩率）、卷“风格偏差”（差异度/读者接受度）
        成长轨迹类：主角能力/地位量化记录、技能/道具升级链条、属性面板具体数值变化、势力扩张具体数据、每次升级的触发事件、成长瓶颈及突破方式、成长“社会标记”（地位体现/势力扩张数据）、成长“代价记录”（隐性成本/瓶颈心理描写）、成长“读者预期差”（预测vs实际偏差记录）
        """
        task_goal = task_info.get('goal', '')
        task_id = task_info.get('id', '')
        task_type = task_info.get('task_type', '')
        
        all_queries = []

        # 添加任务目标作为核心检索词
        all_queries.append(task_goal)

        # 添加概览关键词 - 聚焦故事核心元素
        overview_keywords = [
            "故事进展", "情节发展", "主要角色", "关键事件", 
            "角色关系", "冲突", "转折"
        ]
        all_queries.extend(overview_keywords)
        
        # 具体情节关键词 - 针对正文细节
        specific_keywords = [
            "主角当前状态", "当前场景细节", "当前地点细节", "最近3章关键事件", 
            "主角当前持有物", "近期出场配角", "角色对话细节", "短期关系变化"
        ]
        all_queries.extend(specific_keywords)

        # 提取层级信息（卷、幕、章、场景、节拍）- 添加正文标识
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
                all_queries.append(f"第{num}{level_name} 正文")
                # 前后层级上下文（用于获取连续性）
                try:
                    num_int = int(num)
                    if num_int > 1:
                        all_queries.append(f"第{num_int-1}{level_name} 正文")
                    all_queries.append(f"第{num_int+1}{level_name} 正文")
                except:
                    pass
        
        # 提取任务ID相关上下文 - 增加正文标识
        if task_id and '.' in task_id:
            parent_id = '.'.join(task_id.split('.')[:-1])
            all_queries.append(f"任务 {parent_id} 写作内容")
            # 添加当前任务ID
            all_queries.append(f"任务 {task_id} 写作内容")
        
        # 从最新内容提取关键词 - 优先考虑最新内容
        if latest_content and isinstance(latest_content, str):
            content_keywords = self._extract_keywords_from_text(latest_content)
            # 优先添加最新内容的关键词
            all_queries = content_keywords + all_queries

        design_keywords = self._extract_keywords_from_markdown(same_graph_dependent_designs)
        all_queries.extend(design_keywords)
        
        # 合并所有关键词并去重
        final_queries = list(dict.fromkeys(all_queries))

        # 执行检索
        all_results = self.search(final_queries, "story", limit=500)
        
        # 处理结果：按相关性和时间戳排序
        combined_results = []
        seen_contents = set()
        
        # 对结果进行排序，优先最新内容
        sorted_results = sorted(all_results, key=lambda x: x.get('metadata', {}).get('timestamp', ''), reverse=True)
        
        for result in sorted_results:
            memory_content = result.get('memory', '')
            if memory_content and memory_content not in seen_contents:
                # 过滤掉太短的内容
                if len(memory_content) > 100:
                    combined_results.append(memory_content)
                    seen_contents.add(memory_content)
                # 限制结果数量，避免信息过载
                if len(combined_results) >= 1000:
                    break
        
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


    








