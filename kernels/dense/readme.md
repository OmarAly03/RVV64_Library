## Lenet5
### Layer F4 (120 In --> 84 Out)
```bash
# def_args_dense_test="120 84"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== DENSE LAYER [In: 120, Out: 84] ===
Scalar Cycles: 90292
Vector Cycles: 16198
Speedup: 5.57x
[hw-cycles]:           0
[227338] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  1bc05
Wallclock time:   31.666 s
Simulation speed: 3589.62 cycles/s (3.58962 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```

### Layer F5 (84 In --> 10 Out)
```bash
# def_args_dense_test="84 10"
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== DENSE LAYER [In: 84, Out: 10] ===
Scalar Cycles: 7833
Vector Cycles: 1865
Speedup: 4.20x
[hw-cycles]:           0
[32596] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  3faa
Wallclock time:   3.946 s
Simulation speed: 4130.26 cycles/s (4.13026 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
```