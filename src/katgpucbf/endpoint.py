################################################################################
# Copyright (c) 2015-2019, 2002 National Research Foundation (Square Kilometre Array)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################
"""Endpoint utilities.

It's inelegant but we've copied this file here because it's all we use from
katsdptelstate, and we'd prefer not to have to pull it with all its dependencies
in.
"""

import socket
import struct
from typing import Any, Callable, Iterator, List


class Endpoint:
    """A TCP or UDP endpoint consisting of a host and a port.

    Typically the host should be a string (whether a hostname or IP address) and
    the port should be an integer, but users are free to use other conventions.
    """

    def __init__(self, host: Any, port: Any) -> None:
        self.host = host
        self.port = port

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Endpoint) and self.host == other.host and self.port == other.port

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((self.host, self.port))

    def __str__(self) -> str:
        if ":" in self.host:
            # IPv6 address - escape it
            return "[{}]:{}".format(self.host, self.port)
        else:
            return "{}:{}".format(self.host, self.port)

    def __repr__(self) -> str:
        return "Endpoint({!r}, {!r})".format(self.host, self.port)

    def __iter__(self) -> Iterator:
        """Support `tuple(endpoint)` for passing to a socket function."""
        return iter((self.host, self.port))


def endpoint_parser(default_port: Any) -> Callable[[str], Endpoint]:
    """Return a factory function that parses a string.

    The string is either `hostname`, or `hostname:port`, where `port` is an
    integer. IPv6 addresses are written in square brackets (similar to RFC
    2732) to disambiguate the embedded colons.
    """

    def parser(text: str) -> Endpoint:
        port = default_port
        # Find the last :, which should separate the port
        pos = text.rfind(":")
        # If the host starts with a bracket, do not match a : inside the
        # brackets.
        if len(text) and text[0] == "[":
            right = text.find("]")
            if right != -1:
                pos = text.rfind(":", right + 1)
        if pos != -1:
            host = text[:pos]
            port = int(text[pos + 1 :])
        else:
            host = text
        # Strip the []
        if len(host) and host[0] == "[" and host[-1] == "]":
            # Validate the IPv6 address
            host = host[1:-1]
            try:
                socket.inet_pton(socket.AF_INET6, host)
            except OSError as e:
                raise ValueError(str(e))
        return Endpoint(host, port)

    return parser


def endpoint_list_parser(default_port: Any, single_port: bool = False) -> Callable[[str], List[Endpoint]]:
    """Return a factory function that parses a string.

    The string comprises a comma-separated list, each element of which is of
    the form taken by :func:`endpoint_parser`. Optionally, the hostname may be
    followed by `+count`, where `count` is an integer specifying a number of
    sequential IP addresses (in addition to the explicitly named one). This
    variation is only valid with IPv4 addresses.

    If `single_port` is true, then it will reject any list that contains
    more than one distinct port number, as well as an empty list. This allows
    the user to determine a unique port for the list.
    """

    def parser(text: str) -> List[Endpoint]:
        sub_parser = endpoint_parser(default_port)
        parts = text.split(",")
        endpoints = []
        for part in parts:
            endpoint = sub_parser(part.strip())
            pos = endpoint.host.rfind("+")
            if pos != -1:
                start = endpoint.host[:pos]
                count = int(endpoint.host[pos + 1 :])
                if count < 0:
                    raise ValueError("bad count {}".format(count))
                try:
                    start_raw = struct.unpack(">I", socket.inet_aton(start))[0]
                    for i in range(start_raw, start_raw + count + 1):
                        host = socket.inet_ntoa(struct.pack(">I", i))
                        endpoints.append(Endpoint(host, endpoint.port))
                except OSError:
                    raise ValueError("invalid IPv4 address in {}".format(start))
            else:
                endpoints.append(endpoint)
        if single_port:
            if not endpoints:
                raise ValueError("empty list")
            else:
                for endpoint in endpoints:
                    if endpoint.port != endpoints[0].port:
                        raise ValueError("all endpoints must use the same port")
        return endpoints

    return parser
