#!/usr/bin/env python3
from recursive.agent.prompts.base import PromptTemplate
from recursive.agent.prompts.base import prompt_register


DESIGN_PRINCIPLES = """
# 设计原则

## 基本要求
- 逻辑严谨：时间线连贯，人物动机一致，世界观自洽，因果关系明确
- 避免套路：基于真实人性构建冲突，设置意外转折，融合创新元素
- 通俗易懂：使用日常语言，具象描述，逻辑清晰

## 创意激发
- 头脑风暴：鼓励自由联想，产生多样化的创意点子
- 逆向思维：从结果出发，反推可能的起因和过程
- 创新融合：结合不同领域的元素，创造独特的故事背景

## 动机与冲突分析
- 动机分析：深入挖掘人物、事件或决策的内在动机，确保行为合理且引人共鸣
- 冲突构建：构建复杂的利益、价值观或情感冲突，增加任务的张力
- 意外转折：设置出人意料的情节转折，保持读者的兴趣

## 结构层级
- 结构层级划分：全书→卷（视字数可选）→幕→章（每章2000-5000字）→场景→节拍→段落
- 边界限制：只设计当前层级及直接上下文，禁止涉及未规划的下级细节
- 字数分配：均衡分配，总和等于父任务字数
- 单元要素：层级、序号、标题、字数、详细规划

## 任务吸引力增强
- 节奏控制：合理安排任务节奏，确保情节紧凑且有起伏
- 悬念设置：在关键节点设置悬念，引导读者继续阅读
- 元素发展：设计角色、情节或主题的成长弧线，让读者产生代入感

## 爆款网文设计原则
- 强冲突驱动：每章设置2-4个冲突点，采用"3+1"模型（3个强冲突 + 1个核心悬念）
- 情绪爽点设计：明确主角的核心目标，设计包含实力显摆、打脸反转、绝地逢生等爽点
- 快节奏推进：确保3章完成打脸，5章出现新地图，10章开启主线副本
- 人物塑造：设计极致人设，如黑莲花事业脑女主、高智商反社会男主，并规划角色成长弧光
- 钩子设计：每章植入2-3个钩子，确保300字出现主角危机，800字揭露金手指，1500字设悬念

## 平台合规
- 内容健康向上，避免敏感内容，符合网络小说平台规范
""".strip()


@prompt_register.register_module()
class StoryReasonerZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色定位
你是专业的故事设计师，负责为已确定的写作任务提供具体、可操作的设计方案。

{DESIGN_PRINCIPLES}

# 输出要求
- 直接输出设计结果，不要有多余解释，使用markdown格式
- 结果必须用 `<result></result>` 标签包裹

<result>
[设计结果]
</result>
""".strip()

        content_template = """
# 创作目标与进度
## 终极目标
<root_question>
{to_run_root_question}
</root_question>

## 已完成内容概要
<story_context>
{to_run_mem0_content}
</story_context>

## 最新章节（接续点）
<article_latest>
{to_run_article_latest}
</article_latest>

# 规划层次与依赖
## 整体写作计划
<full_plan>
{to_run_full_plan}
</full_plan>

## 上级设计蓝图
<outer_graph_dependent>
{to_run_outer_graph_dependent}
</outer_graph_dependent>

## 同级已完成设计
<same_graph_dependent>
{to_run_same_graph_dependent}
</same_graph_dependent>

# 当前待设计任务
<target_task>
{to_run_task}
</target_task>

请基于以上信息，为当前任务提供通俗易懂、具体可操作的设计方案：
""".strip()

        super().__init__(system_message, content_template)


@prompt_register.register_module()
class StoryReasonerAggregateZh(PromptTemplate):
    def __init__(self) -> None:
        system_message = f"""
# 角色定位
你是专业的设计整合专家，负责将多个设计成果整合为统一、易懂的最终方案。

# 整合原则
- 去重合并：消除重复内容，保留最优设计
- 冲突解决：识别矛盾之处，选择最符合整体目标的方案
- 逻辑统一：确保整合后的设计逻辑连贯、层次清晰
- 完整性检查：补充缺失要素，确保设计完整可执行

{DESIGN_PRINCIPLES}

# 输出要求
- 直接输出整合后的设计结果，不要有多余解释，使用markdown格式
- 结果必须用 `<result></result>` 标签包裹

<result>
[设计结果]
</result>
""".strip()

        content_template = """
# 创作目标与进度
## 终极目标
<root_question>
{to_run_root_question}
</root_question>

## 已完成内容概要
<story_context>
{to_run_mem0_content}
</story_context>

## 最新章节（接续点）
<article_latest>
{to_run_article_latest}
</article_latest>

# 规划层次与依赖
## 整体写作计划
<full_plan>
{to_run_full_plan}
</full_plan>

## 上级设计蓝图
<outer_graph_dependent>
{to_run_outer_graph_dependent}
</outer_graph_dependent>

## 同级已完成设计
<same_graph_dependent>
{to_run_same_graph_dependent}
</same_graph_dependent>

# 您需要整合和完善的设计结果
<final_aggregate>
{to_run_final_aggregate}
</final_aggregate>

# 当前待设计任务
<target_task>
{to_run_task}
</target_task>

请将以上设计结果整合为简洁易懂的最终方案：
""".strip()
        
        super().__init__(system_message, content_template)