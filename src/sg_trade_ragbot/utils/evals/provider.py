from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.parser import ingestion
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput


async def call_api(prompt, options, context):
    """
    Promptfoo entrypoint, the function must be called call_api
    """
    model_name = options.get('config').get('model_name')
    local = options.get('config').get('local')

    ground_truth = vars.get("ground_truth")

    ingestion.run()

    agent = get_naive_agent(model_name, local)

    response = await agent.run(prompt)

    structured_response = response.get_pyadantic_model(RAGToolOutput)

    return {
        'output': structured_response.answer,
        'retrievals': structured_response.retrievals,
        'ground_truth': ground_truth,
        'metadata': f"{'local' if local else 'remote'}-{model_name}"
    }
