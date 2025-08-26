from typing import Dict, List, Tuple, Union
from recursive.executor.agents import ActionExecutor
from recursive.executor.schema import ActionReturn, ActionStatusCode, AgentReturn, get_json_schema
from recursive.executor.agents.base_agent import BaseAgent
from recursive.utils.file_io import make_mappings
from pprint import pprint
import json
import requests
from recursive.executor.actions.register import executor_register
from recursive.llm.litellm_proxy import llm_client
from loguru import logger
from recursive.utils.file_io import parse_hierarchy_tags_result
from recursive.agent.prompts.base import prompt_register

# The Chinese prompts for ReAct
 
ALL_SYSTEM_MESSAGE_MAPPINGS, ALL_PROMPT_MAPPINGS = make_mappings(__file__)


@executor_register.register_module()
class SearchAgent(BaseAgent):
    """An implementation of ReAct (https://arxiv.org/abs/2210.03629)

    Args:
        llm (BaseModel or BaseAPIModel): a LLM service which can chat
            and act as backend.
        action_executor (ActionExecutor): an action executor to manage
            all actions and their response.
        protocol (ReActProtocol): a wrapper to generate prompt and
            parse the response from LLM / actions.
        max_turn (int): the maximum number of trails for LLM to generate
            plans that can be successfully parsed by ReAct protocol.
            Defaults to 4.
    """

    def __init__(self,
                 prompt_version,
                 action_executor: ActionExecutor,
                 protocol = None,
                 max_turn: int = 10,
                 action_memory = True,
                 remove_history = True,
                 has_search_targets = False,
                 parse_arg_dict = {}) -> None:
        self.message_constructor = prompt_register.module_dict[prompt_version]()
        self.max_turn = max_turn
        self.force_step = '你需要基于历史消息返回一个最终结果'
        self.action_memory = action_memory
        self.remove_history = remove_history
        self.has_search_targets = has_search_targets
        self.parse_arg_dict = parse_arg_dict
        self.action_memory_format = "<turn={{turn}}>\n{}\n</turn>".format(
            "\n".join(["<{tag}>\n{{{key}}}\n</{tag}>".format(tag=tkey[0], key=key) for key, tkey in self.parse_arg_dict.items()])
        )
        self.search_think_format = "<turn={{turn}}>\n{}\n</turn>".format(
            "\n".join(["<{tag}>\n{{{key}}}\n</{tag}>".format(tag=tkey[0], key=key) for key, tkey in self.parse_arg_dict.items() if key != "search_querys"])
        )
        
        # print(self.action_memory_format)
        
        # print("\n\n")
        
        # print(self.search_think_format)
        # exit()
        
        super().__init__(action_executor=action_executor, protocol=None)

    def parse(
        self,
        message
    ):
        """Parse the action returns in a ReAct format.

        Args:
            message (str): The response from LLM with ReAct format.
            action_executor (ActionExecutor): Action executor to
                provide no_action/finish_action name.

        Returns:
            tuple: the return value is a tuple contains:
                - thought (str): contain LLM thought of the current step.
                - action (str): contain action scheduled by LLM.
                - action_input (str): contain the required action input
                    for current action.
        """
        output = {key: parse_hierarchy_tags_result(message, tk) for key, tk in self.parse_arg_dict.items()}
        try:
            output["search_querys"] = json.loads(output["search_querys"])
        except Exception as e:
            logger.error("Parse search querys error with output dict: {}".format(output))
            output["search_querys"] = output["search_querys"].replace("'", '"')
            output["search_querys"] = json.loads(output["search_querys"])
        return output
    
    def format(self,
               chat_history: List[Dict],
               message: List[Dict],
               context_historys: List[Dict],
               action_executor: ActionExecutor,
               turn: int,
               to_run_target_write_tasks: str,
               to_run_root_question: str,
               to_run_outer_write_task: str,
               today_date: str,
               force_stop: bool = False):
        """Generate the ReAct format prompt.

        Args:
            chat_history (List[Dict]): The history log in previous runs.
            inner_step (List[Dict]): The log in the current run.
            action_executor (ActionExecutor): the action manager to
                execute actions.
            force_stop (boolean): whether force the agent to give responses
                under pre-defined turns.

        Returns:
            Dict[Dict]: ReAct format prompt and other parameter.
        """
        
        # makeup tool executoation
        formatted = []
        system_message = self.message_constructor.construct_system_message()
        if system_message.strip() != "":
            formatted.append(dict(role='system', content=system_message))
        formatted += chat_history
        # build inner history
        
        action_memory = []
        inner_call_history = []
        for turn_info in context_historys:
            if not self.has_search_targets:
                action_memory.append(self.action_memory_format.format(**turn_info))
            else:
                action_memory.append(
                    "<turn = {}>\n<observation>\n{}\n</observation>\n<planning_and_think>\n{}\n</planning_and_think>\n<search_tasks_and_querys>\n{}\n</search_tasks_and_querys>\n</turn>".format(
                        turn_info["turn"],
                        turn_info["observation"],
                        turn_info["think"],
                        turn_info["search_tasks_and_querys"],
                    )
                )
            inner_call_history.append({
                "role": "assistant",
                "content": turn_info["response"],
            })
            
            # Safely handle the case where tool_result might be None
            tool_result_content = "No result available due to an error" 
            if turn_info.get("tool_result") and isinstance(turn_info["tool_result"], dict):
                tool_result_content = turn_info["tool_result"].get("result", "No result available")
            
            inner_call_history.append({
                "role": "user",
                "content": tool_result_content,
            })
        action_memory = "\n\n".join(action_memory)
        if not self.remove_history:
            formatted.extend(inner_call_history[:-1])
        
        # Safely handle the tool_result that might be None
        tool_result = "null"
        if len(context_historys) > 0:
            if context_historys[-1].get("tool_result") and isinstance(context_historys[-1]["tool_result"], dict):
                tool_result = context_historys[-1]["tool_result"].get("result", "No result available")
            else:
                tool_result = "No result available due to an error"
                
        final_message = self.message_constructor.construct_prompt(
            to_run_question = message[0]["content"],
            to_run_root_question = to_run_root_question,
            to_run_outer_write_task = to_run_outer_write_task,
            to_run_action_history = action_memory,
            to_run_tool_result = tool_result,
            to_run_turn = turn,
            to_run_target_write_tasks = to_run_target_write_tasks,
            today_date = today_date
        )
        formatted.append(dict(role="user", content=final_message))
        
        ret = {
            "message": formatted,
        }
        return ret
    
    def chat(self, message: Union[str, dict, List[dict]],
             global_start_index, to_run_target_write_tasks, 
             to_run_root_question, to_run_outer_write_task,
             today_date,
             **kwargs) -> AgentReturn:
        # assert isinstance(message, str)       
        if isinstance(message, str):
            inner_history = [dict(role='user', content=message)]
        elif isinstance(message, dict):
            inner_history = [message]
        elif isinstance(message, list):
            inner_history = message[:]
        else:
            raise TypeError(f'unsupported type: {type(message)}')
        offset = len(inner_history)
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        
        # action_memorys = []
        context_historys = []
        return_info = []
        for turn in range(0, self.max_turn):
            prompt = self.format(
                chat_history=[],
                message=inner_history,
                context_historys=context_historys,
                action_executor=self._action_executor,
                force_stop=(turn == self.max_turn - 1),
                to_run_target_write_tasks = to_run_target_write_tasks,
                to_run_root_question = to_run_root_question,
                to_run_outer_write_task = to_run_outer_write_task,
                today_date = today_date,
                turn = turn)

            for key in list(prompt.keys()):
                if key != "message":
                    kwargs[key] = prompt.pop(key)
            
            cnt = 0
            while cnt < 100:
                response = llm_client.call_reasoning(messages = prompt["message"], **kwargs)[0]["message"]["content"]
                try:
                    parsed_resp = self.parse(response)
                except Exception as e:
                    import traceback
                    logger.info("Meet Exception when parse response: {}, error is : {}".format(response, traceback.format_exc()))
                    cnt += 1
                    continue
                break
                    
            current_turn_info = {"turn": turn}
            current_turn_info.update(parsed_resp)
            current_turn_info["response"] = response
            return_info.append(current_turn_info)
            context_historys.append(current_turn_info)
            
            logger.info("Output: \n{}".format(response))
            
            # Finish
            if len(current_turn_info["search_querys"]) == 0:
                break
        
            action = "BingBrowser.full_pipeline_search"
            action_input = {"query_list": current_turn_info["search_querys"],
                            "user_question": message,
                            "think": self.search_think_format.format(**current_turn_info),
                            "global_start_index": global_start_index}
            logger.info("Do Action {}, param: {}".format(
                action, action_input
            ))
            action_return: ActionReturn = self._action_executor(
                action, action_input)
            # print("action_return:\n{}".format(action_return))
            agent_return.actions.append(action_return)
            # print(agent_return, flush=True)
            current_turn_info["tool_result"] = action_return.result
            
            # Update global start index
            try:
                if current_turn_info["tool_result"] and "web_pages" in current_turn_info["tool_result"]:
                    page_cnt = len(current_turn_info["tool_result"]["web_pages"])
                    global_start_index += page_cnt
                else:
                    # No web pages found or tool result doesn't have web_pages
                    page_cnt = 0
                    logger.warning("No web_pages found in tool_result")
            except (TypeError, KeyError) as e:
                # Handle cases where tool_result is None or doesn't have the expected structure
                page_cnt = 0
                logger.warning(f"Error processing tool result: {e}")
            # print(type(current_turn_info["tool_result"]))
            # logger.info("TOOL Result: \n{}".format(action_return.result["result"]))
        else:
            agent_return.response = default_response
        agent_return.inner_steps = inner_history[offset:]
        
        # makeup all results
        results = []
        for turn_idx, current_turn_info in enumerate(return_info[:-1]):
            turn_result = current_turn_info["tool_result"]
            if turn_result:
                result = {
                    "turn": current_turn_info["turn"],
                    "web_pages": turn_result["web_pages"], # = web_pages (selected)
                    "xml_format_result": turn_result["result"], # = xml concat web pages
                    "observation": return_info[turn_idx+1]["observation"],
                }
                results.append(result)
        
        return {
            "ori": return_info,
            "result": results
        }


