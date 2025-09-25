## Neural Execution Machine

#### Gradient framework

TBD:
- 如何让 int 表示的 <code>Instruction</code> 可微
- 如何让 <code>Instruction</code> 的 bit-to-decimal 过程可微
![framework](./images/IMG_1588.jpg)

#### Gradient-free framework
- 将 <code>Register</code>, <code>Instruction</code> 等的表示方法转为bit表示，而不是索引 -> 用于 bit flip
    
    e.g instruction shape: (B,L,prog_maxlen,16)
- 通过 <code>SampleTool</code> 来采样 candidate sets， 所有 set 并行计算，找出 loss 最小的set 设为 *target set*，*target set* 作为 label 指导 <code>Register</code>, <code>Instruction</code>等的生成
- 两种 loss 分别指导不同部分的梯度更新
    
    ##### <code>final loss</code>: 指导 lm-head，register2hidden layer
    ##### <code>assist loss</code>: 指导 一开始的 proj layer
    ##### 中间部分 non-differentible
- Sample Strategies

    1. <code>GroupSample</code> 指定 group 数目， 每一 group 中 所有指令 random flip n个bit
    2. <code>FieldSample</code> 分 field (e.g. op、R) 翻转， 每个field被选择的概率以及对应的flip_num 由loss来决定， 当一个field的loss占比小于0.05，则不考虑翻转
        - bit确实再向我们假设的target逼近，但是离真实想要的输出（以add r0 r1 r2为例）越来越远  如何跳出？
    3. 

![framework](./images/IMG_1591.jpg)



#### New framework
枚举的搜索空间还是比较大，难以靠有限的枚举数量来找到正确的学习方向
- 放弃对于riscv类型的“硬编码”，转为各种指令的probability distribution
- 对于每一行指令，最终结果是各类指令的加权和
- 设置可微寄存器，用于贴合现实程序中效果
- 设置gate机制来控制 计算信息融入hidden state的量
- op1 和 op2 也不是一步parse到实数，而是l位k bit distributions

![framework](./images/IMG_1602.jpg)

