# Sample repo: shop analytics

A tiny fictional pipeline used by the tests and the quick-start demo.

## Layers

Bronze holds raw order events, silver keeps the latest version of each order, and gold
aggregates revenue per store per day.

## Late events

Events older than the watermark (6 hours by default) go to `shop.quarantine.late_orders`
instead of silver.
