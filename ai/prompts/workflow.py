BASE_WORKFLOW_CONSTRUCTION_PROMPT = """Find relevant commands in the provided command library JSON to fulfill the users request.
When selecting commands, be sure to read the command descriptions to determine if there are any command pre-requisites or post-requisites.

User Request:
{query}

Command Library:
{command_library}

Format Instructions:
{format_instructions}
"""

BASE_WORKFLOW_PLANNING_PROMPT = """Find relevant commands in the provided command library JSON to fulfill the users request.
When selecting commands, be sure to read the command descriptions to determine if there are any command pre-requisites or post-requisites.
Finally, order those commands in the correct order (including any commands which fulfill pre-requisites and post-requisites) in a way that can accomplish the users request.

Only return a numbered list of commands and relevant information such as parameters needed to run commands or varaibles to save off

User Request:
{query}

Command Library:
{command_library}
"""