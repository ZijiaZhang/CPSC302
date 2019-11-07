

# CPSC302

- [Errror](#errror)
  * [Absolute Error](#absolute-error)
  * [Relative Error](#relative-error)
- [Source of Error](#source-of-error)
  * [Cancelation Error](#cancelation-error)
- [Algorithm Properties](#algorithm-properties)
  * [Accuracy](#accuracy)
  * [Efficiency](#efficiency)
  * [Robustness](#robustness)
- [Stable and Unstable Algorithm](#stable-and-unstable-algorithm)
- [Floating Point Systems](#floating-point-systems)
  * [Binary](#binary)
  * [Floating point with fix number t:](#floating-point-with-fix-number-t-)
  * [General floating point system](#general-floating-point-system)
- [Solving a function with one Variable](#solving-a-function-with-one-variable)
  * [Bisection Method](#bisection-method)
    + [Pro](#pro)
    + [Counts](#counts)
    + [Method:](#method-)
  * [Fixed point Iteration](#fixed-point-iteration)
    + [Method](#method)
    + [Rate of convergence](#rate-of-convergence)
  * [Newton's Method](#newton-s-method)
    + [Cons](#cons)
    + [Pros](#pros)
    + [Speed of convergence.](#speed-of-convergence)
  * [Secant Method](#secant-method)
    + [Convergence](#convergence)
- [Minimizing Function in one Variable](#minimizing-function-in-one-variable)
- [Solve for linear problem](#solve-for-linear-problem)
  * [Backward substitution.](#backward-substitution)
    + [Constrains:](#constrains-)
    + [Algotrithm:](#algotrithm-)
  * [Forward substitution](#forward-substitution)
    + [Constrains](#constrains)
  * [Gaussian elimination](#gaussian-elimination)
    + [Idea:](#idea-)
    + [Algorithm](#algorithm)
    + [Cost](#cost)
  * [LU decomposition](#lu-decomposition)
    + [Application/ Usage](#application--usage)


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
$fl(x) = \pm*1.\bar{d_1}\bar{d_2} \dots \bar{d_t} \times 2^e$

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

$|x_{k+1} - x^* | \leq M |x_k-x^* |^2$


### Secant Method

We know that
$f'(x_k) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$ 

So we define Secant iteration.

$x_{k+1} = x_{k} - \frac{f(x_k)(x_k-x_{k-1})}{f(x_k)-f(x_{k-1})}$

So we need $x_0$ and $x_1$ to start.

#### Convergence
This is superlinear convergence.
Faset than simple fixed point iteration not as fast as Newton's iteration.

$|x_{k+1} - x^* | \leq \rho_k|x_k-x^* |$

## Minimizing Function in one Variable

1. Find all critical points in the range, that f'(x) = 0
2. Find min points that f''(x) > 0.
3. Compare the values of f(x) and choose the minimal.


## Solve for linear problem

Ax = b

### Backward substitution.
#### Constrains: 
When A is an upper triangular matrix. i.e. all element below the main diagnal are zero: $a_{ij} = 0, \forall i>j$

#### Algotrithm: 
```
for k=n:-1:1
x(k) = ( b(k) - A(k,k+1:n) * x(k+1:n) ) / A(k,k);
end
```
 
 ### Forward substitution
#### Constrains
A is a lower triangular matrix.
$a_{ij} = 0, \forall i<j$

```
for k=1:n
x(k) = (b(k) - A(k,1:k-1) * x(1:k-1) ) / A(k,k);
end
```

### Gaussian elimination

#### Idea:
Use elementary transformation to transform A to a upper triangular matrix.

$MAx = Mb$
Where MA is a upper triangular matrix.

#### Algorithm
```
for k=1:n-1
for i=k+1:n
l(i,k) = A(i,k)/A(k,k);
for j=k+1:n
A(i,j) = A(i,j) - l(i,k)*A(k,j);
end
b(i) = b(i) - l(i,k)*b(k);
end
end
```

#### Cost
Elimination: 
$\approx 2 \sum_{k=1}^{n-1} (n-k)^2 = \frac{2}{3}n^3+O(n^2)$

Backward substitution:
$\approx 2 \sum_{k=1}^{n-1}(n-k) = 2 \frac{(n-1)n}{2} \in O(n^2)$

### LU decomposition
Notice that $M = M^{(n-1)} \dots M^{(2)} M^{(1)}$
Where $M^{k}$ is a step of elementary operation. 
These are all **Lower triangular matrices** 
Then, matrix M is **Unit Lower Triangular**
We also have $L = M^{-1}$ also a lower triangular matrix.

Then $LU = LMA = A$
#### Application/ Usage
If we have a LU decomposition of A. We will get:
- $A = LU$
- $LUx = b$
- let $y = Ux$

Then:
1. Solve for $Ly = b$ (forward substitution)
2. Solve for $Ux = y$ (backward substitution)

### LU Decomposition with Pivoting Strategies

If the pivot in the Gaussian Elimination is small (which is normal). We will have a big multiplier. Which will magnify the roundoff error.

So at each stage of the  elimination, we will choose the largest one to be our pivot. So we will multiply a permutation matrix P to exchange the rows.

That is: 
$PA = LU$

```
for k = 1:n-1
% todo: find q = row index of relative maximum in column k of matrix A
% todo: interchange rows k and q in A and record this in p (also interchange rows in b)
for i=k+1:n
l(i,k) = A(i,k)/A(k,k);
for j = k+1:n
A(i,j) = A(i,j) - l(i,k)*A(k,j);
end
b(i) = b(i) - l(i,k)*b(k);
end
end
```

$B = M^{(n-1)} P^{(n-1)}\dots M^{(1)} P^{(1)}$ 

$P = P^{(n-1)} \dots P^{(1)}$

$B = L^{-1}P$

For example: 
We have a matrix A:

$U = M^{(2)} P^{(2)} M^{(1)}P^{(1)} A$

We can rewrite this To move P2 and P1 together.

$U = M^{(2)} (P^{(2)}M^{(1)} P^{(2)T}) P^{(2)} P^{(1)} A$

$U = M_{new}^{(2)} M_{new}^{(1)}P_{new}A$
 
Here we merged several steps together as one operation.

Then All elements in the new Ms are negation L.

$L = eye - (M_{new}^{(2)} - eye) - (M_{new}^{(1)} - eye)$

----Here, we first initialize the L with its diagonal of all ones.  Then we can delete all the diagnals in the matrix Ms and only keep those that are under the diagonal which is $M - eye$. 

$P = P^{(2)} P^{(1)}$
