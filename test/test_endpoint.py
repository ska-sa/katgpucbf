################################################################################
# Copyright (c) 2015-2019, 2022 National Research Foundation (Square Kilometre Array)
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
"""Tests for the Endpoint class."""

import pytest

from katgpucbf.endpoint import Endpoint, endpoint_list_parser, endpoint_parser


class TestEndpoint:  # noqa: D101
    def test_str(self) -> None:  # noqa: D102
        assert "test.me:80" == str(Endpoint("test.me", 80))
        assert "[1080::8:800:200C:417A]:12345" == str(Endpoint("1080::8:800:200C:417A", 12345))

    def test_repr(self) -> None:  # noqa: D102
        assert "Endpoint('test.me', 80)" == repr(Endpoint("test.me", 80))

    def test_parser_default_port(self) -> None:  # noqa: D102
        parser = endpoint_parser(1234)
        assert Endpoint("hello", 1234) == parser("hello")
        assert Endpoint("192.168.0.1", 1234) == parser("192.168.0.1")
        assert Endpoint("1080::8:800:200C:417A", 1234) == parser("[1080::8:800:200C:417A]")

    def test_parser_port(self) -> None:  # noqa: D102
        parser = endpoint_parser(1234)
        assert Endpoint("hello", 80) == parser("hello:80")
        assert Endpoint("1080::8:800:200C:417A", 80) == parser("[1080::8:800:200C:417A]:80")

    def test_bad_ipv6(self) -> None:  # noqa: D102
        parser = endpoint_parser(1234)
        with pytest.raises(ValueError):
            parser("[notipv6]:1234")

    def test_iter(self) -> None:  # noqa: D102
        endpoint = Endpoint("hello", 80)
        assert ("hello", 80) == tuple(endpoint)

    def test_eq(self) -> None:  # noqa: D102
        assert Endpoint("hello", 80) == Endpoint("hello", 80)
        assert Endpoint("hello", 80) != Endpoint("hello", 90)
        assert Endpoint("hello", 80) != Endpoint("world", 80)
        assert Endpoint("hello", 80) != "not_an_endpoint"

    def test_hash(self) -> None:  # noqa: D102
        assert hash(Endpoint("hello", 80)) == hash(Endpoint("hello", 80))
        assert hash(Endpoint("hello", 80)) != hash(Endpoint("hello", 90))


class TestEndpointList:  # noqa: D101
    def test_parser(self) -> None:  # noqa: D102
        parser = endpoint_list_parser(1234)
        endpoints = parser("hello:80,world,[1080::8:800:200C:417A],192.168.0.255+4,10.0.255.255+3:60")
        expected = [
            Endpoint("hello", 80),
            Endpoint("world", 1234),
            Endpoint("1080::8:800:200C:417A", 1234),
            Endpoint("192.168.0.255", 1234),
            Endpoint("192.168.1.0", 1234),
            Endpoint("192.168.1.1", 1234),
            Endpoint("192.168.1.2", 1234),
            Endpoint("192.168.1.3", 1234),
            Endpoint("10.0.255.255", 60),
            Endpoint("10.1.0.0", 60),
            Endpoint("10.1.0.1", 60),
            Endpoint("10.1.0.2", 60),
        ]
        assert expected == endpoints

    def test_parser_bad_count(self) -> None:  # noqa: D102
        with pytest.raises(ValueError):
            endpoint_list_parser(1234)("192.168.0.1+-4")

    def test_parser_non_integer_count(self) -> None:  # noqa: D102
        with pytest.raises(ValueError):
            endpoint_list_parser(1234)("192.168.0.1+hello")

    def test_parser_count_without_ipv4(self) -> None:  # noqa: D102
        with pytest.raises(ValueError):
            endpoint_list_parser(1234)("hello.world+4")

    def test_parser_single_port(self) -> None:  # noqa: D102
        parser = endpoint_list_parser(1234, single_port=True)
        endpoints = parser("hello:1234,world")
        expected = [Endpoint("hello", 1234), Endpoint("world", 1234)]
        assert expected == endpoints

    def test_parser_single_port_bad(self) -> None:  # noqa: D102
        with pytest.raises(ValueError):
            endpoint_list_parser(1234, single_port=True)("x:123,y:456")
