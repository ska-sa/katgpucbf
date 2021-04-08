"""
TODO: Write this.

TODO: Add command line parameters
"""

import asyncio
import katxgpu.xbengine_proc_loop


async def async_main() -> None:
    """TODO: Write this."""
    print("Async Main running")
    await asyncio.sleep(1)

    xbengine_proc_loop = katxgpu.xbengine_proc_loop.XBEngineProcessingLoop(
        adc_sample_rate_Hz=1712000000,
        heap_accumulation_threshold=52,
        n_ants=64,
        n_channels_total=32768,
        n_channels_per_stream=32768 // 256,
        n_samples_per_channel=256,
        n_pols=2,
        sample_bits=8,
        channel_offset_value=32768 // 256 * 4,
    )

    xbengine_proc_loop.add_udp_ibv_receiver_transport(
        src_ip="239.10.10.10", src_port=7149, interface_ip="10.100.44.1", comp_vector_affinity=2
    )

    xbengine_proc_loop.add_udp_ibv_sender_transport(
        dest_ip="239.10.10.11", dest_port=7149, interface_ip="10.100.44.1", thread_affinity=2
    )
    await xbengine_proc_loop.run()

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
