# Ladder-Residual-Inference
This repository contains the code for inference benchmarking for the paper [Ladder-residual: parallelism-aware architecture for accelerating large model inference with communication overlapping](https://arxiv.org/abs/2501.06589).

If you are interested in training the Ladder Residual models, you can find the training code in [dolomite-engine](https://github.com/IBM/dolomite-engine).

## Ladder Redisual
Tensor Parallelism (TP) is commonly used to partition a large language models (LLMs) across multiple accelerators to reduce the memory load and computation time during training and inference. However, TP is communication bound and thus requires fast interconnects (e.g., NVLink for NVIDIA GPUs) between the devices. However, these fast interconnects are only available on high-end datacenter GPUs. Even in the presence of these fast interconnects, the TP communication is often a significant bottleneck and thus limits the gains that can be achieved by increasing the number of accelerators.

To mitigate this issue, we propose Ladder Residual: a simple architectural modification compatible to all residual-based models that enable straightforward overlapping, effectively hiding the latency of communication. Our insight is that in addition to rather than solely relying on systems-level optimizations, we propose re-architecting the model to separate communication from computation. In this parallel way, the model can continue processing input data even as communication tasks run in the background.

For a Transformer model (Llama-3.1-8B), applying Ladder Residual to all its layers achieves 29% end-to-end wall clock speed up at inference time with TP world size of 8 devices. We refer to such model as the Ladder Transformer. We train a 1B and 3B Ladder Transformer from scratch and observe comparable performance to a standard dense transformer baseline. We also conduct adaptation experiments for our approach and show that it’s possible to adapt parts of the Llama-3.1 8B model with minimal accuracy degradation by retraining on only 3B tokens. 

We further explore an advanced architectural variant that eliminates communication altogether, enabling fast LLM inference on systems lacking high-speed interconnects.

Ladder Residual Transformer (Ladder Transformer) is a decoder-based LLM architecture that allows overlapping of computation with communication for inference via model architecture modification. The proposed approach doesn't require any custom kernels making the method easily scalable and applicable to different hardware architectures and ML frameworks.

![image](./assets/architecture.png)

## Usage
To run the code, you can install this repository and run one of the benchmarking scripts (70B) as follows:
```shell
pip install -e .
sh scripts/throughput-70B.sh
```
for multi-node benchmarking, you can use the following command:
```shell
sh scripts/throughput-405B.sh <rank>
```
where `<rank>` is the rank of the current node, and please make sure to update master_addr and master_port in the script.

## Acknowledgement

This repository is based on [gpt-fast](https://github.com/pytorch-labs/gpt-fast) and runs completely with PyTorch compile. The model architecture in this repository is based on the Llama architecture.
