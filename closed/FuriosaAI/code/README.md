# Furiosa AI's Inference Accelerator: Renegade

This document describes steps to run Furiosa AI's Renegade inference accelerator in general.
Renegade is a system of an accelerator chip and corresponding software stack.
First, this system takes a pretrained TFLite model and compile the model
to make it loadable into the chip. Once the model is loaded,
a user can feed data (e.g., images) from the host DRAM to the on-chip memory and commence an inference task. The software stack supports the above steps.
The following instructions show how to use the Renegade system.

## Compile
Furiosa AI provides a model compiler, `dgc`, which takes a pretrained TFLite model and produces another model ready to be loaded into a Renegade chip. Detailed descriptions are available in each benchmark (e.g., `mobilenet` and `ssd-small`) directory.

## Execution
Once a compiled model is ready, a user can load the model to the chip.
The software stack includes device driver, runtime libraries, and daemon program to access the chip. Specific instructions to use these software facilities are described in each benchmark directory.

The following list shows all the binaries used for our submission.

- dgc -- internal compiler

- iredit -- IR editor

- libdg.so -- runtime library

- libnux.so -- runtime library

- libnpu.so -- runtime library

- npud -- runtime daemon

- xdma.ko -- device driver


Hash values of each of the binaries are as follows:

- dgc, SHA256=a8432e7dc734d7b4f168c5c5a6d857639fdaf695c69324841c376c470621cefc

- iredit, SHA256=a393df4cfefbc89d14636237ad488d6ae6af65f63af0ed9b31184fd4d2cfe783

- libdg.so, SHA256=4e1d6f393f2b8317b84d538daff610fdba0dc93a6667bedc70fc8f183101af2e

- libnux.so, SHA256=34689b809bc6ce93d4e7c24be6ccdef8b17fd4d9ba0e4c3fec82e2dfc42295e4

- libnpu.so, SHA256=551d5161db24c2be0ca1c6ac0179faef178df9e85dcd2e81ed8105000995b3ef

- npud, SHA256=cc6ea96797b999a2b7c5ee2b5425a0fdb583c53b82259f8476f47b390e4463cb

- xdma.ko, SHA256=6060b17d23417b19dcb45507fe9550bb40160e2554867e329b2cf10023666935
