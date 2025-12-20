## Image Installation
- Download the Ubuntu 24.04.3 LTS (Noble Numbat) RISC-V for QEMU from [HERE](https://canonical-ubuntu-hardware-support.readthedocs-hosted.com/boards/how-to/qemu-riscv/)
- Install the prerequisites
	```bash
	sudo apt update
	sudo apt install opensbi qemu-system-riscv64 qemu-efi-riscv64 u-boot-qemu
	```
- Unpack the image:
	```bash
	xz -dk ubuntu-*-preinstalled-server-riscv64.img.xz
	```

- To run the image with qemu-system
	```bash
	qemu-system-riscv64 \
	 -machine virt \
	 -cpu rv64,v=true \
	 -m <RAMSIZE> \
	 -smp <NO.CPU_CORES> \
	 -nographic \
	 -kernel /usr/lib/u-boot/qemu-riscv64_smode/uboot.elf \
	 -device virtio-net-device,netdev=net0 \
	 -netdev user,id=net0 \
	 -drive file=<PATH_TO_UBUNTU_IMG>,format=raw,if=virtio
	 ```
