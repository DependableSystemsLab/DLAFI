# DLAFI
Software-Based Fault Injection for Permanent Faults in DLAs

# ISSRE 2025 Artifacts Evaluation Guide

This README file describes the structure of this project, our fault injection (FI) tools, benchmarks, instructions to build the tool, and, finally, instructions to reproduce the experiments reported in our paper.

## Artifacts Description
### DLAFI - Our FI framework

DLAFI (Deep Learning Accelerator Fault Injector) is a cross layer fault injection framework that supports fault injection at three levels: 1)Hardawre simulations (RTL/Chisel) 2) Low level software (LLVM-IR) and 3)Application level (pytroch)

The goal of DLAFI is to show how slow RTL-level FI can be used to bring mircoarchitecture awareness to software by running a series of small microbenchmarks in hw simulations). We then show that why low level softare is a better to choice compare to application level FI due to its fine granularity capabilty to modification.

#### More Information on Tools build upon (Optional - Not Required for AE)
[LLTFI Documentation](https://github.com/DependableSystemsLab/LLTFI/blob/ISSRE23_AE/README_LLTFI.md)\
[Gemmini Documentation](https://github.com/ucb-bar/gemmini/blob/master/README.md)\



### Benchmarks Used in this Work

### Project Structure


## Environment Setup
The following are the recommended system configurations required to reproduce our  experiments.\\

**CPU:** Intel I5/I7/I9, or AMD Ryzen Threadripper \
**Operation System:**   Ubuntu 20.04 \
**Endianess:**   Little Endian (must)\
**Bit:**   64-bit \
**RAM:**   at least 16 GB (Building verilator for cycle-accurate hardware siulations requires a decent amount of RAM) \
**SSD/HDD:**   at least 100 GB (Required to download our benchmarks and run experiments) \
**GPU:** Not Needed.

**Note**:-  Our experiments were executed on 10 cluster nodes with Intel Xeon E5-2667 @ 2.10GHz (30 cores each), running in parallel without GPU support. However, the good news is that using a different experimental environment **should not affect** the experimental results as long as your environment adheres to the above-mentioned constraints.


## Getting Started
### Installing DLAFI along with its dependencies
We made a docker container with every dependencies pre-installed.


### Obtaining Benchmarks
