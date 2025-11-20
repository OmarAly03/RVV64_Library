appname = relu
input_size ?= "84 1"

build:
	cd .. && make bin/$(appname) def_args_$(appname)=$(input_size)

build_spike:
	cd .. && make bin/$(appname).spike

run_spike:
	cd .. && make spike-run-$(appname)

run_hardware:
	cd ../../hardware && app=$(appname) make simv 

