init:
	git config core.hooksPath .githooks

test:
	cargo test --release

run-examples:
	for eg in `ls ./examples/*.rs | xargs basename --suffix=.rs`; do \
		cargo run --release --example $$eg; \
	done
