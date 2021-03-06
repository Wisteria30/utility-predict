NAME=util-pred
ARG=bash

build:
	docker build -t $(NAME) -f docker/Dockerfile ./docker
run:
	docker run -id --name $(NAME) -v $(PWD):/work -u $(shell id -u):$(shell id -g) -w /work --gpus all $(NAME) $(ARG)
exec:
	docker exec -i $(NAME) $(ARG)
up:
	if [ "$(shell docker ps -af name=$(NAME) | wc -l | tr -d " ")" = "1" ]; then \
		docker build -t $(NAME) -f docker/Dockerfile ./docker ; \
		docker run -itd --name $(NAME) -v $(PWD):/work -u $(id -u):$(id -g) -w /work --gpus all $(NAME) ; \
	elif [ "$(shell docker ps -f name=$(NAME) | wc -l)" = "1" ]; then \
		docker start $(NAME) ; \
	fi
	docker exec -it $(NAME) bash
up-no-gpu:
	if [ "$(shell docker ps -af name=$(NAME) | wc -l | tr -d " ")" = "1" ]; then \
		docker build -t $(NAME) -f docker/Dockerfile ./docker ; \
		docker run -itd --name $(NAME) -v $(PWD):/work -u $(id -u):$(id -g) -w /work $(NAME) ; \
	elif [ "$(shell docker ps -f name=$(NAME) | wc -l | tr -d " ")" = "1" ]; then \
		docker start $(NAME) ; \
	fi
	docker exec -it $(NAME) bash

down:
	-docker stop $(NAME)
	-docker rm $(NAME)

stop:
	docker stop $(NAME)

.PHONY: clean
clean:
	-docker stop $(NAME)
	-docker rm $(NAME)
	-docker rmi $(NAME)
ps:
	docker ps
test:
	docker exec $(NAME) pytest
