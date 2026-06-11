# Quickstart


## Shuffle Basics

`rapidsmpf` is designed as a service that plugs into other libraries. This means
it isn't typically used as a standalone library, and is expected to operate in
some larger runtime.

## Streaming Engine

The Python streaming API exposes {term}`Actor`, {term}`Channel`, and message primitives
for downstream libraries that need to assemble their own pipelines. See
{doc}`api` for the available classes and functions.
