.PHONY: build push run bash test deploy reboot_vm prepare

TAG=deepdriveio/deepdrive_zero
#SSH=gcloud compute ssh deepdrive-problem-coordinator-5
#SERVER_CONTAINER_NAME=klt-deepdrive-problem-coordinator-lnaf

BUILD_ARGS=--network=host -t $(TAG) -f Dockerfile .

build:
	rm -rf ./docker/repos || echo
	git clone --depth 1 $(SPINNINGUP_DIR) ./docker/repos/spinningup
	git clone --depth 1 . ./docker/repos/deepdrive-zero
	cd docker && docker build $(BUILD_ARGS)
	rm -rf ./docker/repos

bash:
	docker run -it $(TAG) bash

pwd:
	echo $(DEEPDRIVE_ZERO_DIR)
