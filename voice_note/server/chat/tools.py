TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "update_chat_settings",
      "description": "Update chat behavior when the user explicitly asks you to change how you respond, such as your style, role, or whether you should think more deliberately.",
      "parameters": {
        "type": "object",
        "properties": {
          "system_prompt": {
            "type": "string",
            "description": "Additional persistent instruction describing how you should behave from now on. Describe the requested style or role, but do not repeat the fixed voice-output constraints."
          },
          "thinking_enabled": {
            "type": "boolean",
            "description": "Whether to enable more deliberate internal thinking. Only change this if the user explicitly asks for it."
          },
        }
      },
      "required": []
    }
  }
]
