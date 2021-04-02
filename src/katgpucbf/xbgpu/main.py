"""TODO: Write this."""

import asyncio


async def async_main() -> None:
    """TODO: Write this."""
    print("Async Main running")
    await asyncio.sleep(1)
    print("Async Main Complete")


def main() -> None:
    """TODO: Write this."""
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(async_main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


if __name__ == "__main__":
    print("Running main program")
    main()
    print("Running main program")
