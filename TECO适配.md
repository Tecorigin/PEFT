# PEFT

## 1. 功能概述
参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法旨在以较低的计算和存储成本，对大型预训练模型进行有效的适应性调整。通过仅微调模型中的一小部分参数而非全部参数，PEFT技术能够大幅降低资源消耗，同时保持与全参数微调相媲美的性能表现。该方法特别适用于需要将大规模预训练模型快速、高效地适配到特定下游任务的应用场景中。PEFT与Transformers库集成，便于模型训练和推理；结合Diffusers管理不同的适配器，以及利用Accelerate实现分布式训练和推理，使得即便是非常庞大的模型也能高效运行。

- 参考实现：
    ```
    url=https://github.com/Tecorigin/PEFT
    ```

## 2. 使用步骤

运行以下命令,clone代码仓库

    git clone https://github.com/Tecorigin/PEFT.git
    cd PEFT

### 2.1 安装peft


    pip install -e .


### 2.2 安装transformers

    cd transformers
    pip install -e .
    cd ..

### 2.3 安装deepspeed
teco适配的deepspeed库，请联系太初内部人员


    cd deepspeed
    rm -rf build *.egg-info dist
    DS_BUILD_CPU_ADAM=1 pip install .
