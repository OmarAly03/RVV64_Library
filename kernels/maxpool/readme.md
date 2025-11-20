## Lenet5
### Pool 1
```bash
## def_args_maxpool = "6 28 28 2 2"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== MAXPOOL BENCHMARK ===
Shape: [6, 28, 28] Pool: 2x2 Stride: 2
Scalar Cycles: 96343
Vector Cycles: 19109
[hw-cycles]:           0
[253612] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  1ef56
Wallclock time:   30.6 s
Simulation speed: 4143.99 cycles/s (4.14399 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```

### Pool 2
```bash
# def_args_maxpool = "16 10 10 2 2"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== MAXPOOL BENCHMARK ===
Shape: [16, 10, 10] Pool: 2x2 Stride: 2
Scalar Cycles: 35617
Vector Cycles: 12273
[hw-cycles]:           0
[112972] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  dca6
Wallclock time:   13.013 s
Simulation speed: 4340.74 cycles/s (4.34074 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```