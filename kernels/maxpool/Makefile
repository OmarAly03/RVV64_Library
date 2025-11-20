appname = maxpool
input_size ?= "6 28 28 2 2"

build:
	cd .. && make bin/$(appname) def_args_$(appname)=$(input_size)

build_spike:
	cd .. && make bin/$(appname).spike

run_spike:
	cd .. && make spike-run-$(appname)

run_hardware:
	cd ../../hardware && app=$(appname) make simv 

clean:
	rm -f ./*.o ./*.spike ./kernel/*.o ./kernel/*.spike
