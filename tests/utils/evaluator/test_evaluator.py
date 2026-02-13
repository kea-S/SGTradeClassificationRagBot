import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sg_trade_ragbot.utils.evals.evaluator import call_api


@pytest.mark.asyncio
async def test_call_api_with_gpt4o_and_local_false_returns_output_not_none():
    prompt = "Live sheep, pure-bred and breeding"
    options = {"config": {"model_name": "gpt-4o", "local": False}}

    # create a fake agent whose run coroutine returns a non-None value
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(return_value="fake model response")

    # Patch both ingestion.run (to avoid heavy IO/indexing) and get_naive_agent
    with patch("sg_trade_ragbot.utils.evals.evaluator.ingestion") as mock_ingestion, \
         patch("sg_trade_ragbot.utils.evals.evaluator.get_naive_agent", return_value=fake_agent):
        mock_ingestion.run = MagicMock(return_value=None)

        result = await call_api(prompt, options, context=None)

    assert "output" in result
    assert result["output"] is not None
