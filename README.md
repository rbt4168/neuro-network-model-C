# neuro-network-model-C
Model is Artificial Neural Network (ANN)

And It's Math Theory

## Main Function

### Activation function (Sigmoid)
$$σ(x) = \frac{1}{1+e^{-x}}$$

$$σ'(x) = σ(x)(1-σ(x))$$

### Error function
$$E(x,T) = \frac{1}{2}(x-T)^2$$

$$\frac{\partial E(x,T)}{\partial x} = x-T$$

## Propagation Function

### Definition
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



### Front propagation
Use $v,w$ to calculate $u$ : 
$$u_{ij} = \sum_{k=1}^{m_i}v_{(i-1)k}w_{(i-1)kj}$$
Use $u$ to calculate $v$ : 
$$v_{ij}=σ(u_{ij})$$

### Back propagation

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
Fix with gradient descendent (GD) : 
$α$ : alpha factor (aka η, learning rate)
$$w_{ijk} \leftarrow w_{ijk} + αδ_{ijk}$$
