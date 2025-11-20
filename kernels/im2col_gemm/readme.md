```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D IM2COL GEMM ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]
Calculating...
Cycles: 720415
Calculating...
Cycles: 225891
[hw-cycles]:           0
[1910322] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  e9319
Wallclock time:   195.482 s
Simulation speed: 4886.18 cycles/s (4.88618 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]

--- Direct Scalar ---
Cycles: 1010905
[hw-cycles]:           0
[2036294] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  f8923
Wallclock time:   196.552 s
Simulation speed: 5180.04 cycles/s (5.18004 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]

--- Direct Vector (M1) ---
Cycles: 1345877

--- Direct Vector (M8) ---
[2716158] %Warning: ara_tb_verilator.sv:47: TOP.ara_tb_verilator: Core Test *** FAILED *** (tohost = 2)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  14b8ff
Wallclock time:   311.835 s
Simulation speed: 4355.12 cycles/s (4.35512 kHz)
make[1]: *** [Makefile:233: simv] Error 2
make[1]: Leaving directory '/home/omar/ara/hardware'
make: *** [Makefile:14: run_hardware] Error 2
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]

--- Direct Vector (M8) ---
Cycles: 66618
[hw-cycles]:           0
[147678] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  1206f
Wallclock time:   18.377 s
Simulation speed: 4018.01 cycles/s (4.01801 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]

--- Direct Vector (M8) ---
Cycles: 66622

--- Im2Col + GEMM Vector (M8) ---
Cycles: 225382
[hw-cycles]:           0
[602950] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  499a3
Wallclock time:   64.778 s
Simulation speed: 4653.97 cycles/s (4.65397 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 16, 16], Kern: [4, 4, 3, 3]

--- Im2Col + GEMM Vector (M8) ---
Cycles: 53054
[hw-cycles]:           0
[120896] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  ec20
Wallclock time:   14.588 s
Simulation speed: 4143.68 cycles/s (4.14368 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 32, 32], Kern: [4, 4, 3, 3]

--- Direct Vector (M8) ---
Cycles: 156138

--- Im2Col + GEMM Vector (M8) ---
Cycles: 163025
[hw-cycles]:           0
[657140] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  5037a
Wallclock time:   88.834 s
Simulation speed: 3698.7 cycles/s (3.6987 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [32, 16, 16], Kern: [32, 32, 3, 3]

--- Direct Vector (M8) ---
Cycles: 3944132

--- Im2Col + GEMM Vector (M8) ---
Cycles: 1647613
[hw-cycles]:           0
[11202628] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  557822
Wallclock time:   1569.16 s
Simulation speed: 3569.63 cycles/s (3.56963 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

```bash
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [4, 32, 32], Kern: [8, 4, 3, 3]

--- Direct Vector (M8) ---
Cycles: 291509

--- Im2Col + GEMM Vector (M8) ---
Cycles: 122702
[hw-cycles]:           0
[847254] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  676cb
Wallclock time:   98.3 s
Simulation speed: 4309.53 cycles/s (4.30953 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```

## Lenet5
### C1
```bash
# Args: InCh=1, InH=32, InW=32, OutCh=6, KH=5, KW=5, StrideH=1, StrideW=1, Pad=0
# def_args_im2col_gemm="1 32 32 6 5 5 1 1 0"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [1, 32, 32], Kern: [6, 1, 5, 5]

--- Direct Vector (M8) ---
Cycles: 157576

--- Im2Col + GEMM Vector (M8) ---
Cycles: 143638
[hw-cycles]:           0
[621290] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  4bd75
Wallclock time:   92.996 s
Simulation speed: 3340.41 cycles/s (3.34041 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```

### C2
```bash
# Args: InCh=6, InH=14, InW=14, OutCh=16, KH=5, KW=5, Stride=1, Pad=0
# def_args_im2col_gemm="6 14 14 16 5 5 1 1 0"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [6, 14, 14], Kern: [16, 6, 5, 5]

--- Direct Vector (M8) ---
Cycles: 743329

--- Im2Col + GEMM Vector (M8) ---
Cycles: 335202
[hw-cycles]:           0
[2176038] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  109a13
Wallclock time:   326.644 s
Simulation speed: 3330.9 cycles/s (3.3309 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```

### C3
```bash
# Args: InCh=16, InH=5, InW=5, OutCh=120, KH=5, KW=5, Stride=1, Pad=0
# def_args_im2col_gemm="16 5 5 120 5 5 1 1 0"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [16, 5, 5], Kern: [120, 16, 5, 5]

--- Direct Vector (M8) ---
Cycles: 3022385
[hw-cycles]:           0
[6059566] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  2e3b17
Wallclock time:   712.51 s
Simulation speed: 4252.27 cycles/s (4.25227 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'

Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== CONV2D BENCHMARK SUITE ===
In: [16, 5, 5], Kern: [120, 16, 5, 5]

--- Im2Col + GEMM Vector (M8) ---
Cycles: 1353234
[hw-cycles]:           0
[2721738] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /home/omar/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  14c3e5
Wallclock time:   387.637 s
Simulation speed: 3510.68 cycles/s (3.51068 kHz)
make[1]: Leaving directory '/home/omar/ara/hardware'
```