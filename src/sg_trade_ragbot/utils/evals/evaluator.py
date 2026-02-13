from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.parser import ingestion


async def call_api(prompt, options, context):
    """
    Promptfoo entrypoint, the function must be called call_api
    """
    model_name = options.get('config').get('model_name')
    local = options.get('config').get('local')

    ingestion.run()

    agent = get_naive_agent(model_name, local)

    result = await agent.run(prompt)

    return {
        'output': str(result)
    }
