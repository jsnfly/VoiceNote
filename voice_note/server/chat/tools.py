MINIMAL_MODE_SYSTEM_PROMPT = """You are a note taking assistant currently. This means you listen to the notes the user
is taking and acknowledge them. A simple 'Mhm, understood.' or 'Yes, I see.' is sufficient. However, if the user asks
a question, or wants to know something you answer the question normally.""".replace("\n", " ")

TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "enable_thinking_mode",
      "description": "Enable or disable thinking mode.",
      "parameters": {
        "type": "object",
        "properties": {
          "enable": {
            "type": "boolean",
            "description": "Boolean indicating whether to enable (true) or disable (false) thinking mode."
          },
        }
      },
      "required": [
        "enable"
      ]
    }
  },
  {
    "type": "function",
    "function": {
      "name": "enable_minimal_mode",
      "description": "Enable or disable minimal mode.",
      "parameters": {
        "type": "object",
        "properties": {
          "enable": {
            "type": "boolean",
            "description": "Boolean indicating whether to enable (true) or disable (false) minimal mode."
          },
        }
      },
      "required": [
        "enable"
      ]
    }
  }
]
