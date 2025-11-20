#!/usr/bin/env python3
# Copyright 2022 ETH Zurich and University of Bologna.
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Matteo Perotti
# Adapted for ReLU by Gemini

# G = ReLU(A) with A=[MxN], G=[MxN]
# arg1, arg2: M, N

import numpy as np
import sys

def emit(name, array, alignment='8'):
  """Prints assembly directives to emit a global array."""
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  # Print data as 4-byte words in little-endian hex
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      s += "%02x" % bs[i+3-n]
    print("    .word 0x%s" % s)

############
## SCRIPT ##
############

if len(sys.argv) == 3:
  M = int(sys.argv[1])
  N = int(sys.argv[2])
  dtype = np.float32
else:
  print("Error. Give me two arguments: M, N.")
  print("G = ReLU(A) with A=[MxN], G=[MxN]")
  sys.exit()

# Input data (random, scaled from -1.0 to 1.0)
A = (np.random.rand(M, N) * 2 - 1).astype(dtype)
# Output placeholder
C = np.zeros([M, N], dtype=dtype)
# Golden result
# G = np.maximum(A, 0).astype(dtype)

# Create the file content
print(".section .data,\"aw\",@progbits")
emit("M", np.array(M, dtype=np.uint64))
emit("N", np.array(N, dtype=np.uint64))
emit("a", A, 'NR_LANES*4') # Input
emit("c", C, 'NR_LANES*4') # Output placeholder
# emit("g", G, 'NR_LANES*4') # Golden reference