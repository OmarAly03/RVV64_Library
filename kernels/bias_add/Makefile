appname = bias_add
input_size ?= "1 120 1 1"

build:
	cd .. && make bin/$(appname) def_args_$(appname)=$(input_size)

build_spike:
	cd .. && make bin/$(appname).spike

run_spike:
	cd .. && make spike-run-$(appname)

run_hardware:
	cd ../../hardware && app=$(appname) make simv 

