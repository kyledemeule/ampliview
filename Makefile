SHELL = /bin/bash

build:
	docker build -t ampliview .

bash:
	docker run -it -p 5000:5000 -v $(shell pwd):/ampliview ampliview /bin/bash

server:
	docker run -it -p 5000:5000 -v $(shell pwd):/ampliview --env FLASK_APP=hello.py ampliview flask run --host=0.0.0.0
