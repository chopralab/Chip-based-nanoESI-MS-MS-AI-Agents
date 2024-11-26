BASE_OBJECTIVE_FUNCTION_CONSTRUCTOR_PROMPT = '''Create an objective function for the optimization provided workflow.

First, identify which parameters in the provided workflow will be optimized.

Second, generate order_kwargs such that it correlates positions of the input tensor to the appropriate
varaibles in the workflow which need to be optimized. 

Finally, determine what command outputs should be saved off to assess the fitness of the optimization.

Workflow:
{info_workflow}

Format Instructions:
{format_instructions}
'''