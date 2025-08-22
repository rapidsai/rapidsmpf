# Background

## Shuffle Statistics

Shuffles can be configured to collect statistics, which can help you understand the performance of the system.
This table gives an overview of the different statistics collected.

| Name | Type | Description |
| --- | --- | --- |
| `spill-bytes-device-to-host` | int | The size in bytes of data moved from device to host when spilling data. |
| `spill-time-device-to-host` | float | The duration of the device to host spill. The unit is platform dependent. |
| `spill-bytes-host-to-device` | int | The size in bytes of data moved from host to device when unspilling data. |
| `spill-time-host-to-device` | float | The duration of the host to device spill. The unit is platform dependent. |
| `spill-bytes-recv-to-host` | int | The size in bytes of data received into host memory on one node from some other node. |
| `shuffle-payload-send` | int | The size in bytes of data transferred from a node (including locally, from a node to itself). |
| `shuffle-payload-recv` | int | The size in bytes of data transferred to a node (including locally, from a node to itself). |
| `event-loop-total` | float | The duration of a Shuffler's event loop iteration. The unit is platform dependent. |
| `event-loop-metadata-send` | float | The duration of sending metadata from one node to another. The unit is platform dependent. |
| `event-loop-metadata-recv` | float | The duration of receiving any outstanding metadata messages from other nodes. The unit is platform dependent. |
| `event-loop-post-incoming-chunk-recv` | float | The duration of posting receives for any incoming chunks from other nodes. The unit is platform dependent. |
| `event-loop-init-gpu-data-send` | float | The duration of receiving ready-for-data messages and initiating data send operations. The duration of the actual data transfer is not captured by this statistic. The unit is platform dependent. |
| `event-loop-check-future-finish` | float | The duration spent checking if any data has finished being sent. The unit is platform dependent. |

Statistics are available in both C++ and [Python](#api-statistics).
