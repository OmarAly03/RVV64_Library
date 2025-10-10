.PHONY: all compile save run

CORE_PATH = ../../core/vicuna

OBJS = "main.o src/rvv_maxpool.o"

all: compile save run

compile:
	make -f $(CORE_PATH)/sw/Makefile PROG=main OBJ=$(OBJS) COMPILER=gcc

save:
	@echo "main.vmem" > progs.txt

run:
	make -s -f $(CORE_PATH)/sim/Makefile PROG_PATHS=progs.txt CORE=cv32e40x

clean:
	@make -f $(CORE_PATH)/sw/Makefile clean
