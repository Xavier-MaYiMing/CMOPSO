### CMOPSO: A competitive mechanism-based multi-objective particle swarm optimizer

##### Reference: Coello C,  Pulido G T,  Lechuga M S. Zhang X, Zheng X, Cheng R, et al. A competitive mechanism based multi-objective particle swarm optimizer with fast convergence[J]. Information Sciences, 2018, 427: 63-76.

The CMOPSO belongs to the category of multi-objective evolutionary algorithms (MOEAs).

| Variables    | Meaning                                              |
| ------------ | ---------------------------------------------------- |
| npop         | Population size                                      |
| iter         | Iteration number                                     |
| lb           | Lower bound                                          |
| ub           | Upper bound                                          |
| pm           | Mutation probability (default = 1)                   |
| eta_m        | Perturbance factor distribution index (default = 20) |
| gamma        | Elite particle size (default = 10)                   |
| dim          | Dimension                                            |
| pos          | Position                                             |
| objs         | Objectives                                           |
| nobj         | Objective number                                     |
| vmax         | Maximum velocity                                     |
| vmin         | Minimum velocity                                     |
| vel          | Velocity                                             |
| leaders      | Elite particles                                      |
| leaders_objs | The objective of elite particles                     |
| off_pos      | Offspring position                                   |
| off_vel      | Offspring velocity                                   |
| off_objs     | Offspring objectives                                 |
| pfs          | Pareto fronts                                        |
| rank         | Pareto rank                                          |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 200, np.array([0] * 10), np.array([1] * 10))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/CMOPSO/Pareto front.png)

```python
Iteration 20 completed.
Iteration 40 completed.
Iteration 60 completed.
Iteration 80 completed.
Iteration 100 completed.
Iteration 120 completed.
Iteration 140 completed.
Iteration 160 completed.
Iteration 180 completed.
Iteration 200 completed.
```

