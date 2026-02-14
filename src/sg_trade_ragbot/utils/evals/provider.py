from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.parser import ingestion
from sg_trade_ragbot.utils.pydantic_models.models import RAGToolOutput


async def call_api(prompt, options, context):
    """
    Promptfoo entrypoint, the function must be called call_api
    """
    model_name = options.get('config').get('model_name')
    local = options.get('config').get('local')

    ground_truth = options.get("ground_truth")

    ingestion.run()

    agent = get_naive_agent(model_name, local)

    response = await agent.run(prompt)

    structured_response = response.get_pydantic_model(RAGToolOutput)

    retrievals_missing = False
    if not structured_response.retrievals:
        serialised_retrievals = []
        retrievals_missing = True
    else:
        serialised_retrievals = [r.model_dump() for r in structured_response.retrievals]

    metadata = {
        retrievals_missing: retrievals_missing,
        local: local,
        model_name: model_name,
    }

    return {
        'output': structured_response.answer,
        'retrievals': serialised_retrievals,
        'ground_truth': ground_truth,
        'metadata': metadata
    }
