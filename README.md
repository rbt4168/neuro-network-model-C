# Neuro Network Model in C
Model is Artificial Neural Network (ANN).

## Trainning Result
```
[ 9040 / 10000 ] validation. accuracy = 90.400000 %.
Judge matrix
        0       1       2       3       4       5       6       7       8       9
0       945     0       4       2       2       7       13      2       5       0
1       0       1114    4       5       1       0       3       2       6       0
2       19      7       898     17      19      2       20      19      25      6
3       6       0       34      879     0       40      7       16      18      10
4       2       1       4       0       912     1       14      3       4       41
5       19      1       13      45      19      723     16      14      32      10
6       19      2       11      0       19      12      888     1       6       0
7       0       15      22      4       11      1       0       944     5       26
8       7       10      9       20      15      27      13      14      851     8
9       13      8       4       9       43      10      1       21      14      886
```

## Mathematical Theory

### Main Functions

#### Activation Function (Sigmoid)
$$σ(x) = \frac{1}{1+e^{-x}}$$

$$σ'(x) = σ(x)(1-σ(x))$$

#### Error function
$$E(x,T) = \frac{1}{2}(x-T)^2$$

$$\frac{\partial E(x,T)}{\partial x} = x-T$$

### Propagation Functions

#### Definition
$U_{ij}$ : neuro unit at $i_{th}$ layer $j_{th}$ place (or a structure contained $u_{ij},v_{ij},x_{ij},y_{ij}$)

$u_{ij}$ : store in $U_{ij}$ ,the sum of input value.

$v_{ij}$ : store in $U_{ij}$ ,the output value.

$x_{ij}$ : store in $U_{ij}$ ,the value of $\partial E/\partial u_{ij}$.

$y_{ij}$ : store in $U_{ij}$ ,the value of $\partial E/\partial v_{ij}$.

$G_{ijk}$ : the edge between layer $i$ and $i+1$ connect the $i$ layer's $j_{th}$ neuro unit to $i+1_{th}$ layer's $k_{th}$ neuro unit (or a structure contained $w_{ijk}$, $δ_{ijk}$).

$w_{ijk}$ : store in $G_{ijk}$ ,the weight of the edge.

$δ_{ijk}$ : store in $G_{ijk}$ ,the value of $\partial E/\partial w_{ijk}$.

$m_i$ : the $i_{th}$ layer's neuro unit count.

$n$ : layer count.



#### Front propagation
Use $v,w$ to calculate $u$ : 
$$u_{ij} = \sum_{k=1}^{m_i}v_{(i-1)k}w_{(i-1)kj}$$
Use $u$ to calculate $v$ : 
$$v_{ij}=σ(u_{ij})$$

#### Back propagation

Calculate last layer's $y$ :
$v_{Ti}$ : the $i_{th}$ value of target output
$$y_{ni} = \frac{\partial E}{\partial v_{ni}} = v_{ni}-v_{Ti}$$
Use $u,y$ to calculate $x$ : 
$$x_{ij} = \frac{\partial E}{\partial u_{ij}} = \frac{\partial E}{\partial v_{ij}}\frac{\partial v_{ij}}{\partial u_{ij}} = y_{ij}(\frac{\partial σ(u_{ij})}{\partial u_{ij}}) = y_{ij}σ'(u_{ij}) = y_{ij}σ(u_{ij})(1-σ(u_{ij}))$$
Use $v,x$ to calculate $δ$ : 
$$δ_{ijk} = \frac{\partial E}{\partial w_{ijk}} = \frac{\partial E}{\partial u_{(i+1)k}}\frac{\partial u_{(i+1)k}}{\partial w_{ijk}} = x_{(i+1)k}(\frac{\partial(\sum_{j=1}^{m_{i+1}}v_{ij}w_{ijk})}{\partial w_{ijk}}) = x_{(i+1)k}v_{ij}$$
Use $w,x$ to calculate other layer's $y$ :
$$y_{ij} = \frac {\partial E}{\partial v_{ij}} =  \sum_{k=1}^{m_{i+1}}\frac{\partial u_{(i+1)k}}{\partial v_{ij}}\frac{\partial E}{\partial u_{(i+1)k}} = \sum_{k=1}^{m_{i+1}}\frac{\partial(\sum_{j=1}^{m_{i+1}}v_{ij}w_{ijk})}{\partial v_{ij}}\frac{\partial E}{\partial u_{(i+1)k}} =  \sum_{k=1}^{m_{i+1}}w_{ijk}x_{(i+1)k}$$

### Fix weight
Fix with stochastic gradient descendent (SGD) : 

$α$ : alpha factor (aka η, learning rate)

$K$ : batch size.

$$w_{ijk} \leftarrow w_{ijk} + α\frac{1}{K}\sum δ_{ijk}$$

My method is set $α=0.9, K=300$

## Reference
https://en.wikipedia.org/wiki/Stochastic_gradient_descent

https://en.wikipedia.org/wiki/Learning_rate

https://medium.com/uxai/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E9%A6%AC%E6%8B%89%E6%9D%BE-075-%E5%8F%8D%E5%90%91%E5%82%B3%E6%92%AD-backpropagation-f1b612e003df


C is the most beautiful language in the world.

--rbt4168 2022/11/06
