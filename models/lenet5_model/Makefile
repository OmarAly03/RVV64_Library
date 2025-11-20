appname = lenet5_model

build:
	cd .. && make bin/$(appname)

build_spike:
	cd .. && make bin/$(appname).spike

run_spike:
	cd .. && make spike-run-$(appname)

run_hardware:
	cd ../../hardware && app=$(appname) make simv 

clean:
	rm -f ./*.o ./*.spike ./kernel/*.o ./kernel/*.spike
