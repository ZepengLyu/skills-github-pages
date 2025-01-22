---
title: Welcome to my blog
---


备注：因为呈现效果的原因，文中就没有用加粗表示向量了，

# 行列式与子式


余子式 $M_{ij}$ 为去掉 $a_{ij}$ 所在行与列构成的矩阵的行列式

代数余子式（cofactors），为 $C_{ij}=(-1)^{i+j}M_{ij}$

行列式（Determinant）是使用代数余子式来计算的，可以使用任意行去计算行列式，并且结果相同（下式基于 $i$ 行计算）

$$
D(A)=\sum_ja_{ij}C_{ij}
$$

# 共轭的两个数（复数）相乘为实数
即 $\overline{x}\cdot x=r$，其中 $r$ 是一个实数。对于向量 $x$，则有 $x^T\cdot x=r$。 
# （两个复数）乘积的共轭等于共轭的乘积
即 $\overline{x\cdot y}=\overline{x}\cdot\overline{y}$

设 $x=a+bi$，$y=c+di$ 的基本形式分别带入左式和右式即可以完成证明，结果都为 $ac-bd+(ad+bc)i$。就该结果我们可以这样理解，$x$ 中的 $bi$，其实是给 $a$ 增加了一个隐性负效果，使得其在与其他复数相乘的时候，这样的隐形负效果和对方的隐性负效果相乘的时候能够产生显性负效果（即 $ bd $ ）。

# 矩阵的乘法运算与结合律
矩阵最核心的运算是乘法，该如何把握矩阵的乘法呢？

例如 $A\times B\times C$ 的计算结果，对于结果的第（i，k）项，其结构包含着两个加法，
$\sum_j (\sum_h a_{ih}b_{hj} )c_{jk} $ .
对这个两重求和式我们可以观察，在确定了索引（i，k）之后，矩阵 $B$ 的每一项都会参与计算，而矩阵 $A$，$C$ 仅有 $A$ 的第 $i$ 行，$C$ 的$ k $ 列会参与计算。结果其实是基于矩阵 $B$的全部元素（经过修饰）的相加得到的，而修饰则是$A_{i \cdot}$,$C_{\cdot k}$参与的乘积变换，结果的第 (i,k) 项正是由 $j \cdot h$ 项（B矩阵元素个数）相加得到的。



从以上过程我们也可以看出，矩阵的乘法满足结合律。因为 $A$ 和 $C$ 分别作用于行与列。

进一步也可以说，一系列矩阵的乘法中，最前面那个矩阵的行之间保持基本独立，最后面那个矩阵的列之间保持基本独立。



# 实对称矩阵的特征值都是实数

证明过程如下：

$Ax=\lambda x$（ 假设讨论的特征向量为$x$,特征值为$ \lambda $ ）


$A\bar{x}=\bar{\lambda}\bar{x}$

（利用共轭的性质，并且试图分离出 $\bar{\lambda}$ ）->

$x^TA\bar{x}=x^T\bar{\lambda}\bar{x} \ $     

$(Ax)^T\bar{x}=\bar{\lambda}x^T\bar{x}$  

$(\lambda x)^T\bar{x}=\bar{\lambda}x^T\bar{x}$

$\lambda x^T\bar{x}=\bar{\lambda}x^T\bar{x}$ 

由于 $x^T\bar{x}$ 不为0，因此 $\lambda=\bar{\lambda}$


# 非正定矩阵的特征值都是非负的

非正定矩阵是基于矩阵的二次型形式（ $f^T A f$ ）来定义的，即 $f^T A f \ge 0$ 总是成立，无论$f$为何值。

$Ax=\lambda x$（ 假设讨论的特征向量为$x$,特征值为$ \lambda $ ）


$\lambda =\lambda x^Tx=x^T\lambda x=x^TA  x\ge 0$

# 矩阵/向量的求导（偏导）
标量函数对向量求导，全微分 $d f$ 是梯度 $\nabla f$ 与微分向量 $dx$ 的内积。

$$
\frac{\partial \ AB} {\partial x}=\frac{\partial \ A} {\partial x}B+（A\frac{\partial \ B} {\partial x}）^T
$$

雅可比矩阵（向量值函数的一阶导数）
$$
J(f)=\begin{pmatrix}
 \frac{\partial f_1}{\partial x_1} & ... & \frac{\partial f_1}{\partial x_n}  \\
 ... & ... &... \\
 \frac{\partial f_m}{\partial x_1}   &  & \frac{\partial f_m}{\partial x_n} 
\end{pmatrix}
$$
（列是变量维度，行是向量值函数的维度）
# 分块矩阵
矩阵$A$，$B$的相乘可以使用分块矩阵相乘的方式，即如果

$A=\begin{bmatrix}
A_1  &A_2 \\
A_3  &A_4
\end{bmatrix}$，$B=\begin{bmatrix}
B_1  &B_2 \\
B_3  &B_4
\end{bmatrix}$

那么$AB=\begin{bmatrix}
A_1B_1+A_2B_3  &A_1B_2+A_2B_4 \\
A_3B_1+A_4B_3  &A_3B_2+A_4B_4
\end{bmatrix}$

（从矩阵乘法的过程可以看出来）

# 矩阵的特征向量具有正交性

# 矩阵的特征分解（对角化）

# 不是所有矩阵的都具有实特征值与实特征向量
# 矩阵行列式的值等于其特征值的乘积。
# 两个普通向量构成的矩阵的特征值。
对于向量 $u,v$，矩阵 $uv^T$ 的特征值只有一个非零数，而其他都是0。这是因为矩阵 $uv^T$ 的秩为1（可以通过列变换看到只剩一个向量）。此时该非零特征值为矩阵的迹 $Tr(uv^T)$，也即是 $u^Tv$。

# 矩阵的迹等于矩阵特征值的和


# 矩阵的解
对于线性方程组 $AX=b$，其中 $A$ 的维度为 $m\times n$。

根据 $ m$ 与$n$ 的数量关系，解也有不同的类型:

- 当 $m=n$，且矩阵 $A$ 秩为 $m$ 时，方程的解则直接有
$$

X=\begin{pmatrix}
 B^{-1}b  \\
 0 
 \end{pmatrix}
$$
(此时 $B^{-1}$ 就是 $A^{-1}$)
- 当 $ m<n$ 且矩阵A的秩为 $m$ 时，假设基向量矩阵为 $B$（并且位于矩阵 $A$ 的前 $m$ 列），非基向量矩阵设为 $D$(位于矩阵 $A$ 的后 $n-m$ 列)，则方程的解可以表示为

$$
X=\begin{pmatrix}
 B^{-1}b  \\
 0 
\end{pmatrix}
+\begin{pmatrix}
 -B^{-1}DX_D  \\
 X_D 
\end{pmatrix}
$$
（右边第二项的上下两部分的效果产生了抵消，从矩阵的分块运算出发）
- 当 $ m>n$ 时方程不一定存在解，此时问题变为，寻求一个 $x_0$ 使得 $||Ax-b|| $最小，$x_0$ 此时也被称为**最小二乘解**。\
最小二乘解为:
$$
x_0=(A^TA)^{-1}A^Tb
$$

（它正是 $A^TAX=A^Tb$ 计算得来的。此时由于 $\mathrm{rank}(A)=n$， $A^TA$ 非奇异，事实上它们是充分必要关系)
    实际上，对于 $x_0=(A^TA)^{-1}A^Tb$，可以将 $(A^TA)^{-1}A^T$ 视为正交投影算子。

- 如果要求矩阵的解是整数，则应该对矩阵 $A$ 进行限制。实际上，当矩阵 $A$ 是如下定义的幺模矩阵时，所有的解都是整数解

    对于 $m\times n$ 的整数矩阵 $A\in Z^{m\times n}\ (m\le n)$，如果其所有 $m$ 阶非零子式为 $+1$ 或者 $-1$，那么 $A$ 就是**幺模矩阵**。
    
$A$ 是一个幺模矩阵，等价于对于任意基矩阵 $B$，都有$|\det B|=1$，并且正是因为后者的原因，$B^{-1}$是一个整数矩阵。由上面讨论的 $m<n$ 的情况，可知所有解都是整数解。


# Gram-Schmidt 正交化
Gram-Schmidt 正交化是一种从已有向量组生成共轭向量组的方法。
向量的共轭是指，对于某一个对称正定矩阵 $Q$，如果 $p^TQq=0$，那么称向量 $p$ 与 $q$ 共轭。共轭是正交的延伸。

假设从 $p_1,p_2,...,p_n$ 生成关于 $Q$ 共轭的方向 $d_1$,$d_2$,...,$d_n$。令$d_1=p_1$，
$$
d_{k}=p_k-\sum_{i=1}^{k-1}\frac{p_{k-1}^TQp_{k}}{p_{k-1}^TQp_{k-1}}p_{k-1}
$$
(可以用待定系数证明得到)

# Sherman-Morrison公式
Sherman-Morrison公式可以快速获得某一矩阵（被秩1矩阵修饰）的逆。矩阵的逆是一种很麻烦的计算，可以用LU分解，伴随矩阵等方式求解，但是计算量也较大。

矩阵 $A\in R^{ n\times n}$ 非奇异，向量 $u,v\in R^n$ 满足 $1+v^TA^{-1}u\ne 0$，则 
$$
(A+uv^T)^{-1}=A^{-1}-\frac{A^{-1}uv^TA^{-1}}{1+v^TA^{-1}u}
$$
#### 推导过程：
1、首先解出下列方程中的 $x$
$$
(I+uv^T)x=b
$$
得到 
$$
x=(I-\frac{uv^T}{1+v^Tu})b
$$
(这个解的过程还是挺特别的)，这样就得到了Sherman-morrison公式的简单版本（当 $b$ 为 $I$ 时）。

2、对于 $(A+uv^T)^{-1}$，进行变换：
$$
\begin{align}
    (A+uv^T)^{-1}&=[A(I+A^{-1}uv^T)]^{-1}=(I+A^{-1}uv^T)^{-1}A^{-1}\\
    &=(I-\frac{A^{-1}uv^T}{1+v^TA^{-1}u})A^{-1}\\
\end{align}
$$
- - - 
# 矩阵的空间：
### 零空间与值空间 

对于矩阵 $A$，称 $\mathcal R (A):=\{y:y=Ax;\forall x \in R^n\}$ 为 $A$ 的值空间。而称$\mathcal{N}(A):=\{x:Ax=0;\forall x \in R^n\}$ 为 $A$ 的零空间。

对于矩阵 $A$ 的值空间和零空间，它们有如下关系：
$$
 \mathcal R(A)= \mathcal N (A^T)^\bot
$$
(使用集合相互包含的方式证明)

### 正交算子
**正交投影算子**（矩阵） 的定义：对于 $\forall x\in R ^n$，都有$Px\in \mathcal V$且 $x-Px\in \mathcal V^\bot$，则称线性变换 $P$ 是 $\mathcal V$（一般是一个子空间） 上的正交投影算子。

矩阵（线性变换）$P$ 是一个正交算子,当且仅当 $P^2=P=P^T$



对于 $b\in R^n$，要寻找 $b$ 落在子空间 $\mathcal V\subset R^n$ 的正交投影。如果 $\mathcal V$ 在 $\mathcal R(A)$ 中\
（$A$ 为 $m\times n$ 矩阵，且 $m\ge n$，秩为 $n$），则正交投影算子 $P=A(A^TA)^{-1}A^T$.\
其实这个正交投影就是该子空间内最接近 $b$ 的向量，或者说，最小二乘解（通过对平方误差函数求导，令导数为0可以得到）。

如果 $\mathcal V$ 在 $\mathcal N(A)$ 中（$A$ 为 $m\times n$ 矩阵，且 $m\le n$）,则正交投影算子为$P=I-A^T(AA^T)^{-1}A$。\
（这是使用值空间与零空间的性质+最小二乘解的性质得到的）


- - - 
# 奇异值分解（SVD）
对于一个 $m\times n$ 的矩阵，尝试遵循方阵的分析方法将其分解。对于方阵 $A$ 而言，可以分解为 $A=U\Lambda U^T$。对于 $m\times n$ 的矩阵 $A$，尝试分解为 $A=U\Sigma V^T$，此时矩阵 $U$ 的维度为 $m\times m$，而矩阵 $V^T$ 的维度为 $n\times n$，它们都是酉矩阵，$\Sigma$ 是 $m\times n$ 对角矩阵。

如何求得其中的 $U,\Sigma,V^T$呢？

如果我们将矩阵 $A$ 左乘其 $A^T$，即得到 $n\times n$ 矩阵: $A^TA=V\Sigma ^TU^TU\Sigma V^T =V\Sigma^T\Sigma V^T$，由此能够得到 $V^T$。

如果我们将矩阵 $A$ 右乘其 $A^T$，即得到 $m\times m$ 矩阵: $AA^T=U\Sigma V^TV\Sigma^T U^T =U\Sigma\Sigma^T U^T$，由此能够得到 $U$。

在上面的过程中，我们也可以得到 $\Sigma$，它是$AA^T$或者$A^TA$的特征值矩阵的开根号之后的值。

应用：如果将矩阵 $A$ 进行奇异值分解，可能会发现其中的奇异值矩阵 $\Sigma $ 有许多零值（至少有 $\max(m,n)-\min(m,n)$ 个），并且有可能非零值有很多小值，这时候我们就可以尝试得到矩阵 $A$ 的一个近似。

此外，基于SVD我们可以将一个矩阵分为若干个低秩矩阵的组合，方阵的特征分解也会得到类似的组合。

例如对矩阵 $A$ 进行特征分解，得到 $A=U\Lambda U^T$，可以得到 $A=\lambda_1u_1u_1^T+\lambda_2u_2u_2^T...+\lambda_nu_nu_n^T$，其中 
$u_i$ 是 $U$ 的第 $i$ 列。$u_iu_i^T$ 为秩 $1$ 矩阵。

# PCA 主成分分析
主成分分析（Principal Component Analysis）的背景是，对于 $m\times n$ 数据矩阵X，试图对其进行降维，变成一个 $m\times k$ 矩阵。在该降维过程中，应该尽可能让数据点之间的相对关系保持不变，而同时使数据维度 $k$ 尽量可能小。前者通过使变换后数据矩阵每个维度的方差最大实现的，而后者是使数据维度之间近可能正交（不冗余）实现的。

假设 $Y=XP$，那么我们试图降低各维度之间的相关性，这实际上等同于增加各维度之间的差异性，让每列数据之间尽可能区分开来,如果其协方差矩阵 $\bar Y^T\bar Y$ 是一个对角矩阵 （$Y$的数据每列经过去中心化处理之后得到$\widetilde Y$），那么应该是一种较好的方式，这表明各列数据之间没有相关性。
而 $\widetilde Y^T\widetilde Y=P^T\widetilde X^T\widetilde X P$。

对$\widetilde X^T\widetilde X $ 进行特征分解可以得到 $\widetilde X^T\widetilde X=U\Lambda U^T$，那么可以看到，取 $P=U$即可让 $\bar Y^T\bar Y$ 变成一个对角矩阵。但实际上，这并没有起到降维的作用，只起到了正交变换的作用。我们目的是得到的一个$n\times k$的变换矩阵 $P$。实际上，在此处我们只取$U$中的某几个特征向量，它们对应的特征值是较大的。（我们这样取的原因在于，在特征值矩阵 $\Lambda$ 中，可能有很多个较小的非负值，而将这些特征值视为0，对矩阵来说几乎没有影响）。也就是说，我们生成了一个  $\widetilde X^T\widetilde X$ 的一个近似矩阵，$U_k\Lambda_kU_k^T$，其中$U_k$是  $n\times k$ 矩阵，$\Lambda_k$ 是$k\times k$ 矩阵。这样我们就得到了一个$P=U_k$的变换矩阵。

这是基于特征分解的PCA，在上面过程中，$U_k$也是 $\widetilde X $的奇异值分解的右乘矩阵$V^T$的转置$V$。SVD有一些算法能够直接计算得到 $V$，基于这些方法的PCA也称为基于奇异值分解的PCA。

- - -
# 复特征分解
**Hermite 共轭**：根据复数的性质，有 $\bar z^Tz=R=||z||^2$，将 $\bar z^T $ 记为 $z^H$，则有 $z^Hz=R=||z||^2$，$z^H$ 也称为 $z$ 的 Hermite 共轭。
A 的所有特征值的全体叫做 A 的谱
# 相似矩阵
如果存在可逆矩阵$P$，使得
$$
B=P^{-1}AP
$$
则称矩阵 $A,B$ 相似。

因此也可以看到，在特征分解中，特征值矩阵与原矩阵相似。
相似矩阵可以认为是同一个线性变换在不同基下的表示。

- - -
# 行列式
矩阵 $A$ 的行列式除了可以使用展开式计算，还可以借助矩阵 $A$ 的伴随矩阵与逆矩阵进行计算
$$
\mathrm{Adju}(A)=\det A\cdot A^{-1}
$$

# 行列式的导数
