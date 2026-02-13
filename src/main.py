import asyncio
from sg_trade_ragbot.agents.naive_agent import get_naive_agent
from sg_trade_ragbot.parser import ingestion
from sg_trade_ragbot.utils.models.models import REMOTE_LLAMA3


async def main():
    ingestion.run()

    agent = get_naive_agent(REMOTE_LLAMA3, False)

    result = await agent.run()

    return result


if __name__ == "__main__":
    result = asyncio.run(main())

    print(str(result))
