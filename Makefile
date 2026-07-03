.PHONY: check
check: fmt-check lint test

.PHONY: fmt-check
fmt-check:
	cargo fmt --check

.PHONY: lint
lint:
	cargo clippy --all-targets -- -D warnings

.PHONY: test
test:
	cargo test
