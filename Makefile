.PHONY: build push run bash test deploy reboot_vm prepare

TAG=deepdriveio/deepdrive_zero
#SSH=gcloud compute ssh deepdrive-problem-coordinator-5
#SERVER_CONTAINER_NAME=klt-deepdrive-problem-coordinator-lnaf

BUILD_ARGS=--network=host -t $(TAG) -f Dockerfile .
PWD=$(shell pwd)

build:
	echo Please sure you have committed your changes locally.
	read press_when_done_commiting
	rm -rf ./docker/repos || echo
	git clone --depth 1 file:///$(SPINNINGUP_DIR) ./docker/repos/spinningup
	git clone --depth 1 file:///$(PWD) docker/repos/deepdrive-zero
	cd docker && docker build $(BUILD_ARGS)
	rm -rf ./docker/repos

bash:
	docker run -it $(TAG) bash

push:
	docker push $(TAG)