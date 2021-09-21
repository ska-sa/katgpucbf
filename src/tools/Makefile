################################################################################
# Copyright (c) 2020-2021, National Research Foundation (SARAO)
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

CXX = g++
CXXFLAGS = $(shell pkg-config --cflags spead2) -std=c++17 -Wall -Wextra -g -O3 -pthread
LDFLAGS = $(shell pkg-config --static --libs spead2) -lboost_program_options
TARGETS = dsim fsim

all: $(TARGETS)

%: %.cpp
	$(CXX) -o $@ $< $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
