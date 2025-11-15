.PHONY: run-app

run-app: up

create-network:
	@if ! docker network inspect support_agent_net > /dev/null 2>&1; then \
		docker network create support_agent_net; \
		echo "Сеть support_agent_net создана."; \
	else \
		echo "Сеть support_agent_net уже существует."; \
	fi

up: create-network
	docker compose up