# CPSC302
## Errror
### Absolute Error
The absolute error in v approximating u is $|u-v|$.
### Relative Error
The absolute error in v approximating u is $\frac{|u-v|}{|u|}$.

## Source of Error
1. Error in the problems:
	- In the mathematical model
	- In input data
2. Approximation errors
	- Discretization errors
		Arise from discretizations of continuous processes, such as interpolation, differentiation, and integration
	- Convergence errors
	Arise in iterative methods. For instance, nonlinear problems must generally be solved approximately by an iterative process. Such a process would converge to the exact solution in infinitely many iterations, but we cut it off after a finite (hopefully small!) number of such iterations. Iterative methods in fact often arise in linear algebra.
3.  Roundoff errors
	These arise because of the finite precision representation of real numbers on any computer, which affects both data representation and computer arithmetic.
	We cannot avoid round off error accumulation like $E_n \approx c_0nE_0$

### Cancelation Error
When two nearby numbers are subtracted, the relative error is large. That is, if $x\approx y$ then $x-y$ has a large relative error.


## Algorithm Properties

### Accuracy
How accurate the algorithm to the true answer.
### Efficiency
What is the cost of the algorithm. Measured by Big O or rate of convergence.
### Robustness
If the algoritm works for all the cases.
## Stable and Unstable Algorithm

The algorithm is stable if its output is the exact result of a slightly perturbed input.
e.g. An algorithm with recursion that is like $y_i = ky_{i-1}$ with $k>0$ will be unstable.

## Floating Point Systems

### Binary
The representation of x in binary is:

$x = \pm (1+ \frac{d_1}{2} + \frac{d_2}{4} + \dots) \times 2^e$ 
with binary digits $d_i = 0$ or 1 and exponent e.

### Floating point with fix number t:
$fl(x) = \pm*1.\bar{d_1}\bar{d_2} \dots \bar{d_t}) \times 2^e$

To repersent the number in floating point form:
Copy the first t digets. For the remaining degits if it is cloaser to one bigger, add it. Otherwise no.

### General floating point system

Defined by ($\beta,t,L,U$) , where:
- β: base of the number system (for binary, $\beta = 2$ ; for decimal $\beta = 10$, ); 
- t precision (number of digits); 
- L: lower bound on exponent e;
- U: upper bound on exponent e.

## Solving a function with one Variable
### Bisection Method
#### Pro
- Simple 
- Safe, robust
 - Requires only that f be continuous
#### Counts
- Slow
- Hard to generalize to systems of equations

#### Method:
Given a<b such that $f(a)f(b)\lt0$ There must be a root in [a,b]. let $p = \frac{a+b}{2}$ if $f(a)f(p)<0$ set b->p othetwise set a->p


### Fixed point Iteration

Given a problem that has $f(x) = 0$. We can transform it to $g(x)=x$ So that $f(x*) =0$ iff $g(x*)=x*$.

#### Method
We iterate so that $x_{k+1} = g(x_k)$. Start with initial guess $x_0$


#### Rate of convergence
let $\rho$ Be the max $g'(x)$ in range a,b. Then the rate of convergence is $-\log_{10}\rho$
It takes about $k = \frac{1}{rate}$ iterations

If $\rho >0$ there is no gareenteed convergence.

### Newton's Method

#### Cons
- Not so simple 
- Not very safe or robust 
- Requires more than continuity on f

#### Pros
- Fast
- Automatically generalizes to systems of equations


Set $g(x) = x - \frac{f(x)}{f'(x)}$

Do fix point Iteraton.

#### Speed of convergence.

Quadratically convergent. There is a constant M.

$|x_{k+1} - x^*| \leq M |x_k-x^*|^2$


### Secant Method

We know that
$f'(x_k) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$ 

So we define Secant iteration.

$x_{k+1} = x_{k} - \frac{f(x_k)(x_k-x_{k-1})}{f(x_k)-f(x_{k-1})}$

So we need $x_0$ and $x_1$ to start.

#### Convergence
This is superlinear convergence.
Faset than simple fixed point iteration not as fast as Newton's iteration.

$|x_{k+1} - x^*| \leq \rho_k|x_k-x^*|$

## Minimizing Function in one Variable

1. Find all critical points in the range, that f'(x) = 0
2. Find min points that f''(x) > 0.
3. Compare the values of f(x) and choose the minimal.

