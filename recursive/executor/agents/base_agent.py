from recursive.executor.actions import ActionExecutor
from recursive.executor.actions.base_action import BaseAction
from recursive.executor.schema import AgentReturn


class BaseAgent:
    def __init__(self, action_executor: ActionExecutor,
                 protocol: object) -> None:
        self._action_executor = action_executor
        self._protocol = protocol

    def add_action(self, action: BaseAction) -> None:
        """Add an action to the action executor.

        Args:
            action (BaseAction): the action to be added.
        """
        self._action_executor.add_action(action)

    def del_action(self, name: str) -> None:
        """Delete an action from the action executor.

        Args:
            name (str): the name of the action to be deleted.
        """
        self._action_executor.del_action(name)

    def chat(self, message: str, **kwargs) -> AgentReturn:
        raise NotImplementedError
