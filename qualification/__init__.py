"""A few handy things intended for correlator qualification."""
from typing import Union

import aiokatcp
from katsdptelstate.endpoint import Endpoint, endpoint_parser


async def get_sensor_val(client: aiokatcp.Client, sensor_name: str) -> Union[int, float, str]:
    """Get the value of a katcp sensor.

    If the sensor value can't be cast as an int or a float (in that order), the
    value will get returned as a string. This simple implementation ignores the
    actual type advertised by the server.
    """
    _reply, informs = await client.request("sensor-value", sensor_name)

    expected_types = [int, float, str]
    for t in expected_types:
        try:
            return aiokatcp.decode(t, informs[0].arguments[4])
        except ValueError:
            continue


async def get_dsim_endpoint(pc_client: Endpoint, adc_sample_rate: float) -> Endpoint:
    """Get the katcp address for a named product controller from the master."""
    return endpoint_parser(None)(await get_sensor_val(pc_client, f"sim.m800.{int(adc_sample_rate)}.0.port"))


class CorrelatorRemoteControl:
    """A remote control for the correlator created by the fixture."""

    def __init__(self, product_controller_client: aiokatcp.Client, dsim_client: aiokatcp.Client) -> None:
        self.product_controller_client = product_controller_client
        self.dsim_client = dsim_client
