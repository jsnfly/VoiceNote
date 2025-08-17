import json
from typing import Dict, Any, Tuple, Callable


TOOL_PROMPT = """You have access to the following tools. To use a tool, you must respond *only* with a JSON object
inside a <tool_call> block. The JSON should have "name" and "parameters" keys.
Example: <tool_call>{"name": "enable_thinking", "parameters": {"enable": true}}</tool_call>
Available tools:
- Tool `enable_thinking`: Enables or disables the "thinking mode" for generating responses. Parameters: `{"enable": <boolean>}`
- Tool `minimal_mode`: Enters or exits minimal mode. This will start a new conversation. Parameters: `{"enable": <boolean>}`""".replace("\n", " ")

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a helpful, smart and funny assistant talking directly to the user by leveraging
speech-to-text and text-to-speech. So keep your responses concise like in a real conversation and do not use any
spechial characters (including dashes, asteriks and so on) or emojis as they can not be expressed by the
text-to-speech component. {tool_prompt}""".replace("\n", " ")
MINIMAL_MODE_SYSTEM_PROMPT_TEMPLATE = """You are an assistant currently in a 'minimal' mode of operation. This means
you listen to the notes the user is taking and acknowledge them. A simple 'mhm', 'okay', or 'I see' is sufficient.
Only if the user asks a question, you should answer normally. {tool_prompt}""".replace("\n", " ")


class ToolManager:
    def __init__(self):
        self.tools: Dict[str, Callable] = {
            "enable_thinking": self._tool_enable_thinking,
            "minimal_mode": self._tool_minimal_mode,
        }

    def get_default_system_prompt(self) -> str:
        return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(tool_prompt=TOOL_PROMPT)

    @staticmethod
    def _tool_enable_thinking(enable: bool) -> Tuple[Dict[str, Any], str]:
        updates = {"thinking_enabled": bool(enable)}
        mode = "enabled" if updates["thinking_enabled"] else "disabled"
        message = f"Okay, I have {mode} thinking mode."
        return updates, message

    @staticmethod
    def _tool_minimal_mode(enable: bool) -> Tuple[Dict[str, Any], str]:
        if enable:
            template = MINIMAL_MODE_SYSTEM_PROMPT_TEMPLATE
            mode = "entering"
        else:
            template = DEFAULT_SYSTEM_PROMPT_TEMPLATE
            mode = "exiting"

        updates = {
            "system_prompt": template.format(tool_prompt=TOOL_PROMPT),
            "reset_conversation": True
        }
        message = f"Okay, I am {mode} minimal mode."
        return updates, message

    def execute_tool_call(self, tool_call_str: str) -> Tuple[Dict[str, Any], str]:
        try:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call.get("name")
            tool_params = tool_call.get("parameters", {})

            if tool_name in self.tools:
                tool_func = self.tools[tool_name]
                updates, message = tool_func(**tool_params)
                return updates, message
            else:
                return {}, f"Sorry, I don't know the tool '{tool_name}'."
        except json.JSONDecodeError:
            return {}, "Sorry, I received an invalid tool call format."
        except Exception as e:
            return {}, f"An error occurred while executing the tool: {e}"
