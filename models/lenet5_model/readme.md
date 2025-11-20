```bash
omar@omar-Legion-5:/media/omar/983C384F3C382AA0/ara/apps/lenet5_model$ make build
cd .. && make bin/lenet5_model
make[1]: Entering directory '/media/omar/983C384F3C382AA0/ara/apps'
cd lenet5_model && if [ -d script ]; then python3 script/gen_data.py  > data.S ; else touch data.S; fi
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c lenet5_model/data.S -o lenet5_model/data.S.o
chmod +x /media/omar/983C384F3C382AA0/ara/apps/common/script/align_sections.sh
rm -f /media/omar/983C384F3C382AA0/ara/apps/common/link.ld && cp /media/omar/983C384F3C382AA0/ara/apps/common/arch.link.ld /media/omar/983C384F3C382AA0/ara/apps/common/link.ld
/media/omar/983C384F3C382AA0/ara/apps/common/script/align_sections.sh 4 /media/omar/983C384F3C382AA0/ara/apps/common/link.ld
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c common/crt0.S -o common/crt0-llvm.S.o
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c common/printf.c -o common/printf-llvm.c.o
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c common/string.c -o common/string-llvm.c.o
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c common/serial.c -o common/serial-llvm.c.o
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -c common/util.c -o common/util-llvm.c.o
mkdir -p bin/
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -Iinclude -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -ffunction-sections -fdata-sections -std=gnu99 -o bin/lenet5_model lenet5_model/data.S.o lenet5_model/kernel/kernels.cpp.o lenet5_model/main.cpp.o common/crt0-llvm.S.o common/printf-llvm.c.o common/string-llvm.c.o common/serial-llvm.c.o common/util-llvm.c.o -static -nostartfiles -lm -Wl,--gc-sections -T/media/omar/983C384F3C382AA0/ara/apps/common/link.ld
ld.lld: warning: ignoring memory region assignment for non-allocatable section '.comment'
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/llvm-objdump --mattr=v -D bin/lenet5_model > bin/lenet5_model.dump
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/llvm-strip bin/lenet5_model -S --strip-unneeded
rm common/serial-llvm.c.o common/util-llvm.c.o common/printf-llvm.c.o common/crt0-llvm.S.o common/string-llvm.c.o
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/apps'
omar@omar-Legion-5:/media/omar/983C384F3C382AA0/ara/apps/lenet5_model$ make build_spike
cd .. && make bin/lenet5_model.spike
make[1]: Entering directory '/media/omar/983C384F3C382AA0/ara/apps'
cd lenet5_model && if [ -d script ]; then python3 script/gen_data.py  > data.S ; else touch data.S; fi
sed -i s/"li t0, MSTATUS_FS | MSTATUS_XS$"/"li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS"/ /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S
git update-index --assume-unchanged /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -c lenet5_model/data.S -o lenet5_model/data.S.o.spike
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -c /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S -o /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S.o.spike
mkdir -p bin/
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -Iinclude -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -o bin/lenet5_model.spike lenet5_model/data.S.o.spike lenet5_model/kernel/kernels.cpp.o.spike lenet5_model/main.cpp.o.spike /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S.o.spike /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/syscalls.c.o.spike common/util.c.o.spike -static -nostartfiles -lm -nostdlib -T/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/test.ld -Wl,--gc-sections -DSPIKE
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/llvm-objdump --mattr=v -D bin/lenet5_model.spike > bin/lenet5_model.spike.dump
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/apps'
omar@omar-Legion-5:/media/omar/983C384F3C382AA0/ara/apps/lenet5_model$ make run_spike
cd .. && make spike-run-lenet5_model
make[1]: Entering directory '/media/omar/983C384F3C382AA0/ara/apps'
cd lenet5_model && if [ -d script ]; then python3 script/gen_data.py  > data.S ; else touch data.S; fi
sed -i s/"li t0, MSTATUS_FS | MSTATUS_XS$"/"li t0, MSTATUS_FS | MSTATUS_XS | MSTATUS_VS"/ /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S
git update-index --assume-unchanged /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -c lenet5_model/data.S -o lenet5_model/data.S.o.spike
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -c /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S -o /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S.o.spike
mkdir -p bin/
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/clang -Iinclude -march=rv64gcv_zfh -mabi=lp64d -mno-relax -fuse-ld=lld -fno-vectorize -mllvm -scalable-vectorization=off -mllvm -riscv-v-vector-bits-min=0 -mno-implicit-float  -mcmodel=medany -I/media/omar/983C384F3C382AA0/ara/apps/common -O3 -ffast-math -fno-common -fno-builtin-printf  -DNR_LANES=4 -DVLEN=1024 -Wunused-variable -Wall -Wextra -Wno-unused-command-line-argument  -DPREALLOCATE=1 -DSPIKE=1 -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/env -I/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common -ffunction-sections -fdata-sections -std=gnu99 -o bin/lenet5_model.spike lenet5_model/data.S.o.spike lenet5_model/kernel/kernels.cpp.o.spike lenet5_model/main.cpp.o.spike /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/crt.S.o.spike /media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/syscalls.c.o.spike common/util.c.o.spike -static -nostartfiles -lm -nostdlib -T/media/omar/983C384F3C382AA0/ara/apps/riscv-tests/benchmarks/common/test.ld -Wl,--gc-sections -DSPIKE
/media/omar/983C384F3C382AA0/ara/install/riscv-llvm/bin/llvm-objdump --mattr=v -D bin/lenet5_model.spike > bin/lenet5_model.spike.dump
mkdir -p spike_runs

Simulating lenet5_model with SPIKE:

/media/omar/983C384F3C382AA0/ara/install/riscv-isa-sim-mod/bin/spike --isa=rv64gcv_zfh --varch="vlen:1024,elen:64" bin/lenet5_model.spike | tee spike_runs/spike-run-lenet5_model

=== LeNet-5 INFERENCE START ===

[Layer 1] C1 (1->6) + P1
Layer 1 Cycles: 0

[Layer 2] C2 Split (6->16) + P2 + Add
Layer 2 Cycles: 0

[Layer 3] C3 (16->120)
Layer 3 Cycles: 0

[Layer 4] F4 (120->84)
Layer 4 Cycles: 0

[Layer 5] F5 (84->10)
Layer 5 Cycles: 0

[Layer 6] Softmax
Layer 6 Cycles: 0

=== PREDICTION COMPLETE ===
Total Inference Cycles: 0
Class Probabilities:
Class 0: %f
Class 1: %f
Class 2: %f
Class 3: %f
Class 4: %f
Class 5: %f
Class 6: %f
Class 7: %f
Class 8: %f
Class 9: %f
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/apps'
omar@omar-Legion-5:/media/omar/983C384F3C382AA0/ara/apps/lenet5_model$ make run_hardware
cd ../../hardware && app=lenet5_model make simv 
make[1]: Entering directory '/media/omar/983C384F3C382AA0/ara/hardware'
Makefile:83: "Specified QuestaSim version (questa-2021.3) not found in PATH /home/omar/.local/bin:/home/omar/.local/bin:/home/omar/.local/bin:/opt/verilator/bin:/opt/verilator/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin:/opt/riscv/bin:/opt/riscv32/bin:/home/omar/.vscode/extensions/ms-python.debugpy-2025.16.0-linux-x64/bundled/scripts/noConfigScripts:/home/omar/.config/Code/User/globalStorage/github.copilot-chat/debugCommand:/opt/riscv/bin:/opt/riscv32/bin"
build/verilator/Vara_tb_verilator  -l ram,/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model,elf
Program header number 0 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' low is 80000000
Program header number 0 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' high is 800022bd
Program header number 1 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' high is 80042327
Program header number 2 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' high is 8004292f
Program header number 3 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' high is 800eee07
Program header number 4 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' is not of type PT_LOAD; ignoring.
Program header number 5 in `/media/omar/983C384F3C382AA0/ara/apps/bin/lenet5_model' is not of type PT_LOAD; ignoring.
Set `ram TOP.ara_tb_verilator.dut.i_ara_soc.i_dram 10 0x80000000 0x100000 write with offset: 0x0 write with size: 0xeee08
Simulation of Ara
=================


Simulation running, end by pressing CTRL-c.

=== LeNet-5 INFERENCE START ===

[Layer 1] C1 (1->6) + P1
Layer 1 Cycles: 388472

[Layer 2] C2 Split (6->16) + P2 + Add
Layer 2 Cycles: 568765

[Layer 3] C3 (16->120)
Layer 3 Cycles: 1149262

[Layer 4] F4 (120->84)
Layer 4 Cycles: 16529

[Layer 5] F5 (84->10)
Layer 5 Cycles: 1904

[Layer 6] Softmax
Layer 6 Cycles: 1010

=== PREDICTION COMPLETE ===
Total Inference Cycles: 2125942
Class Probabilities:
Class 0: 0.000000
Class 1: 0.000000
Class 2: 0.000000
Class 3: 1.000000
Class 4: 0.000000
Class 5: 0.000000
Class 6: 0.000000
Class 7: 0.000000
Class 8: 0.000000
Class 9: 0.000000
[hw-cycles]:           0
[4311360] -Info: ara_tb_verilator.sv:51: TOP.ara_tb_verilator: Core Test *** SUCCESS *** (tohost = 0)
- /media/omar/983C384F3C382AA0/ara/hardware/tb/ara_tb_verilator.sv:54: Verilog $finish
Received $finish() from Verilog, shutting down simulation.

Simulation statistics
=====================
Executed cycles:  20e4a0
Wallclock time:   504.494 s
Simulation speed: 4272.95 cycles/s (4.27295 kHz)
make[1]: Leaving directory '/media/omar/983C384F3C382AA0/ara/hardware'
omar@omar-Legion-5:/media/omar/983C384F3C382AA0/ara/apps/lenet5_model$ 
```