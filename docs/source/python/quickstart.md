# Quickstart


## Shuffle Basics

`rapidsmpf` is designed as a service that plugs into other libraries. This means
it isn't typically used as a standalone library, and is expected to operate in
some larger runtime.

## Streaming Engine

Basic streaming pipeline example in Python.  In this example we have 3 {term}`Actor`s
in the {term}`Network`: push_to_channel->count_num_rows->pull_from_channel.

*note: push_to_channel/pull_from_channel are convenience functions which simulate scans/writes*

```{literalinclude} ../../../python/rapidsmpf/rapidsmpf/examples/streaming/basic_example.py
:language: python
:lines: 34-
```
