#!/usr/bin/env python3
"""
调试版本的完整测试
"""

import sys
import os
import time
import ssl
import urllib3
from pathlib import Path
from dotenv import load_dotenv





def debug_print(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()








env_path = Path("recursive/api_key.env")
load_dotenv(env_path)



sys.path.insert(0, str(Path(__file__).parent / "recursive"))



try:
    from llm.litellm_proxy import LiteLLMProxy
except Exception as e:
    debug_print(f"❌ 导入失败: {e}")
    sys.exit(1)


try:
    proxy = LiteLLMProxy()
except Exception as e:
    debug_print(f"❌ 初始化失败: {e}")
    sys.exit(1)


model = 'openai/deepseek-ai/DeepSeek-R1-0528'
messages1 = [
    {
        'role': 'system', 
        'content': '# 角色\n你是小说任务复杂度评估专家，精确判定任务是否适合作为原子任务直接执行\n\n输出格式\n- 不要有多余解释：\n<result>\n<atomic_task_determination>\natomic/complex\n</atomic_task_determination>\n</result>'
    }, 
    {
        'role': 'user', 
        'content': '# 创作目标与进度\n## 终极目标\n<root_question>\n10 万字爆款网络小说：背景含极端社会危机与国际激烈博弈，融合科幻、玄幻、修仙。需黄金开篇，爽点密集，冲突反转不断，多线交织、高潮迭起，避同质化套路，紧张 - 释放 - 更紧张循环，小 - 中 - 大悬念递进\n</root_question>\n\n请基于以上信息，判定当前任务的复杂度：'
    }
]
messages2 = [
    {
        'role': 'system', 
        'content': '# 角色\n你是小说任务复杂度评估专家，精确判定任务是否适合作为原子任务直接执行。\n\n# 任务复杂度判定\n\n## 核心原则\n- 原子任务（atomic）：LLM能在单次调用中高质量完成的任务\n- 复杂任务（complex）：需要进一步分解才能保证质量的任务\n\n## 判定标准\n\n## write任务\n- 字数≤2000字 且 有明确依赖（dependency不为空） 且 设计结果详细 → 原子\n- 字数>3000字 或 无依赖（dependency为空） 或 目标宽泛或抽象 或 多个不同类型  → 复杂\n\n## think任务\n- 单一设计要素 且 信息充足 且 范围有限 → 原子\n- 多要素协调 或 系统性规划 或 设计范围大 或 需要分层思考 → 复杂\n\n## 判定原则\n- 确保每个原子任务都能产出高质量结果\n- 平衡效率与质量，避免过度分解\n\n# 输出格式\n- 不要有多余解释：\n<result>\n<atomic_task_determination>\natomic/complex\n</atomic_task_determination>\n</result>'
    }, 
    {
        'role': 'user', 
        'content': '# 创作目标与进度\n## 终极目标\n<root_question>\n10 万字爆款网络小说：背景含极端社会危机（男女对立、道德崩坏、资本垄断、贫富分化、阶级剥削、权力腐败、党同伐异、通胀、生态危机）与国际激烈博弈（两极争霸、合纵连横、地缘博弈），融合科幻、玄幻、修仙。需黄金开篇，爽点密集，冲突反转不断，多线交织、高潮迭起，避同质化套路，紧张 - 释放 - 更紧张循环，小 - 中 - 大悬念递进\n</root_question>\n\n## 已完成内容概要\n<story_context>\n\n</story_context>\n\n## 最新章节（接续点）\n<article_latest>\n\n</article_latest>\n\n# 规划层次与依赖\n## 整体写作计划\n<full_plan>\n{\'id\': \'1\', \'task_type\': \'write\', \'goal\': \'10 万字爆款网络小说：背景含极端社会危机（男女对立、道德崩坏、资本垄断、贫富分化、阶级剥削、权力腐败、党同伐异、通胀、生态危机）与国际激烈博弈（两极争霸、合纵连横、地缘博弈），融合科幻、玄幻、修仙。需黄金开篇，爽点密集，冲突反转不断，多线交织、高潮迭起，避同质化套路，紧张 - 释放 - 更紧张循环，小 - 中 - 大悬念递进\', \'dependency\': [], \'finish\': False, \'is_current_to_plan_task\': True, \'sub_tasks\': []}\n</full_plan>\n\n## 上级设计蓝图\n<outer_graph_dependent>\n\n</outer_graph_dependent>\n\n## 同级已完成设计\n<same_graph_dependent>\n\n</same_graph_dependent>\n\n# 当前待评估任务\n<target_task>\n{"id": "1", "goal": "10 万字爆款网络小说：背景含极端社会危机（男女对立、道德崩坏、资本垄断、贫富分化、阶级剥削、权力腐败、党同伐异、通胀、生态危机）与国际激烈博弈（两极争霸、合纵连横、地缘博弈），融合科幻、玄幻、修仙。需黄金开篇，爽点密集，冲突反转不断，多线交织、高潮迭起，避同质化套路，紧张 - 释放 - 更紧张循环，小 - 中 - 大悬念递进", "task_type": "write", "length": "根据任务要求确定", "dependency": []}\n</target_task>\n\n请基于以上信息，判定当前任务的复杂度：'
    }
]


# # 测试短消息
# debug_print("🧪 测试短消息...")
# try:
#     response = proxy.call(
#         model=model,
#         messages=messages1
#     )
#     if response and len(response) > 0:
#         content = response[0].message.content
#         debug_print(f"📝 短消息响应成功: {len(content)} 字符")
#     else:
#         debug_print("❌ 短消息无响应内容")
        
# except Exception as e:
#     debug_print(f"❌ 短消息调用失败: {e}")

# 测试长消息
debug_print("🧪 测试长消息...")
try:
    response = proxy.call(
        model=model,
        messages=messages2
    )
    if response and len(response) > 0:
        content = response[0].message.content
        debug_print(f"📝 长消息响应成功: {content} ")
    else:
        debug_print("❌ 长消息无响应内容")
        
except TimeoutError:
    debug_print("❌ 长消息调用超时")
except Exception as e:
    debug_print(f"❌ 长消息调用失败: {e}")
    debug_print(f"错误类型: {type(e).__name__}")
    import traceback
    debug_print("详细错误:")
    traceback.print_exc()

debug_print("🏁 测试结束")