from pydantic import ValidationError
import re

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

    try:
        structured_response = RAGToolOutput.model_validate_json(str(response))

        answer = structured_response.answer
        retrievals = [
            item.model_dump(exclude_none=True, mode="json") if hasattr(item, "model_dump") else item
            for item in (structured_response.retrievals or [])
        ]

        metadata = {"local": local, "model_name": model_name}

        return {
            "output": answer,
            "retrievals": retrievals,
            "metadata": metadata,
            "ground_truth": ground_truth,
        }
    except ValidationError as e:
        return {'error': str(e.errors())}
