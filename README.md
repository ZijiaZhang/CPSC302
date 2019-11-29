






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

### Error in the Solution

Say we have a approximate solution $\hat{x}$. We want to get the error it with the real solution x. We will use $\frac{\| x - \hat{x}\|}{\|x\|}$ as the relative error. But we don't know the exact solution $x$. 

#### Approximate the Error.
We have an estimate: $\hat{r} = b - A\hat{x}$ as residual. And relative residual $\frac{\|\hat{r}\|}{\|b\|}$. This is am approximation of the error. If the condition number ($\kappa(A) = \|A\|\|A^{-1}\|$) is small, we will have a good approximation on the relative error.

#### Condition Number
- $\kappa _2(A)$, $\kappa_\infty(A)$. The subscript indicates what norm is used.
- $\kappa(A) \geq 1$
- For Orthogonal matrices $\kappa(Q) = 1$
- $\kappa(A)$ indicate how close A is to singular.
- If A is symmetric positive definite. $\kappa(A) = \frac{\text{Biggest eigenvalue}}{\text{Smallest eigenvalue}}$
- If A is nonsingular $\kappa(A) = \frac{\text{Biggest singular value}}{\text{Smallest singular value}}$

### Cholesky Decomposition

Since A is a symmetric positive definite, We can write $A = GG^T$ Where G is a lower triangular matrix. 

Since A is symmetric positive definite matrix, we can write $A = LU$.

We can rewrite U as: 
$U = D * \widetilde{U} = \begin{pmatrix}  
u_{11} \\  
 & u_{22} \\
 && u_{33} \\
 &&& \ddots \\
 &&&&u_{nn}  
\end{pmatrix} *
\begin{pmatrix}  
1 & \frac{u_{12}}{u_{11}} &  \dots & \dots & \frac{u_{1n}}{u_{11}}\\  
 & 1 & \frac{u_{23}} {u_{22}} &\dots & \frac{u_{2n}}{u_{22}}  \\
 && 1 \\
 &&& \ddots \\
 &&& &1  
\end{pmatrix}$ 

Then we have
$A = LD\widetilde{U}$
Because A is symmetric, we can have $L^T = \widetilde{U}$
$A = LDL^T$
We can see that $G = LD^{1/2}$ 

```
for k=1:n 
A(k,k) = sqrt(A(k,k));
for i=k+1:n 
A(i,k) = A(i,k) / A(k,k);
end 
for j=k+1:n
 for i=j:n
  A(i,j) = A(i,j) - A(i,k) * A(j,k);
 end 
end
end
```

### Banded Matrices

#### GE decomposition for Sparse Matrices

```
for k = 1:n-1
for i=k+1:k+p-1 % fewer iterations
l(i,k) = A(i,k)/A(k,k);
for j = k+1:k+q-1 % fewer iterations
A(i,j) = A(i,j) - l(i,k)*A(k,j);
end
b(i) = b(i) - l(i,k)*b(k);
end
end
```
**Float Count**
$O(n)$

#### GEPP with banded matrices.

- The upper bandwidth of U may increase from q to q + p -1. 
- The matrix L, although it remains unit lower triangular, is no longer tightly banded (although it has the same number of nonzeros per column).

#### Fill-in Issue.

If A is a sparse matrix, after decomposition, the matrix U and L might have more entries than A.
We can use **Permutations and Ordering** to solve the Fill-in Issue.


#### Data fitting
**Linear Data fitting**
Given a set of data points A and value b, we want to find an x that minimize $\|b-Ax\|$
with A size of $m\times n$ and $m>n$

**Basic Algorithm**
To solve $\min_x\|b-Ax\|$ is to solve $A^TAx = A^Tb$
The cost of this method will be mostly $A^Tb$ if $m \gg n$.
This might cause issue when A is most linear dependent columns.

**QR decomposition**
Notice that $\|b-Ax\| = \|r\| = \|Pr\| = \|Pb - PAx\|$ for any Orthogonal Matrix P.
So, find orthogonal matrix $Q = P^T$ that transforms A into upper triangular form

$A = Q \begin{pmatrix} R \\ 0\end{pmatrix}$
To solve the equation, with QR decomposition:

$\|b-Ax\| = \|b - Q \begin{pmatrix} R \\ 0\end{pmatrix}x\| = \|Q^Tb - \begin{pmatrix} R \\ 0\end{pmatrix}x\|$

$\|Q^Tb\| = \begin{pmatrix} c\\ d\end{pmatrix}$ Where d is the corresponding rows of the R, and d are those corresponding to "0".

$\|r\|^2 = \|b-Ax\|^2 = \|c-Rx\|^2 + \|d\|^2$

We can see the solve x in  $Rx = c$, so we can obtain $r = d$.
Solving $Rx = c$ we can use back-substitution.
Flop Count:
$2mn^2 - \frac 2 3n^3$ Roughly double the normal equations.

**Gram-Schmidt orthogonalization**
A way to calculate QR decomposition.

Say a 2 by 2 matrix:

$\begin{pmatrix} a_{11} &a_{12}\\ a_{21} &a_{22}\\ 
a_{31} &a_{32}\\ \end{pmatrix}
 = \begin{pmatrix} q_{11} &q_{12}\\ q_{21} &q_{22}\\ 
q_{31} &q_{32}\\ \end{pmatrix}
\begin{pmatrix} r_{11} &r_{12}\\ 0 &r_{22} \end{pmatrix}$ 
We rewrite in this way:
$(a_1, a_2) = (q_1, q_2) \begin{pmatrix} r_{11} &r_{12}\\ 0 &r_{22} \end{pmatrix}$

So we can have 
$a_1 = q_1*r_{11}$

$a_2 = q_1 *r_{12} + q_2* r_{22}$

Because $q_1$ and $q_2$ are orthonormal, 

$\|q_1\|$ = 1 

We can see that $r_{11} = \|a_1\|$ and $q_1 = a_1/r_{11}$

Because $q_1$ and $q_2$ are orthonormal, and $a_2 = r_{12}q_1 + r_{22}q_2$. We can see $r_{12}$ is the projection of $a_2$ onto $q_1$. so $r_{12} = <a_2,q_1>$. And we come back to the condition calculating the first column.


**Modified Gram-Schmidt orthogonalization**

The classical Gram-Schmidt method becomes unstable when $\kappa(A)$is large.

```
for j=1:n
 q(:,j) = a(:,j);
  for i=1:j-1
   r(i,j) = q(:,j)' * q(:,i);
  end 
 r(j,j) = norm(q(:,j)); 
 q(:,j) = q(:,j)/r(j,j);
end
```

**Householder transformation**
- ***SaveForLater***

**SVD & TSVD decomposition**
- SVD
$A = U\Sigma V^T$ so 
$\|b-Ax\| = \|b - U\Sigma V^T x \| = \|U^Tb-\Sigma V^Tx\| = \|U^Tb-\Sigma y\|$ with $x = Vy$, because U and V are Orthognal matrix.
Denote $U^Tb = z$ we can see that the optimal y is $y_i = z_i/\sigma_i$ for $1 \leq i \leq r$ where r is the rank of A.  This is similar to the QR decomposition method above, that is as stable as the QR decomposition method.

- TSVD
  If we don't know the effective rank r, we would approximate the r. That is: 
  If $\kappa(A)$ is too large we would check for i = n to 1 that the first i makes $\sigma_1/\sigma_i$ with in a certain therathhold.
  We than set $\sigma_{r+1} \dots \sigma_{n}$ to 0.


***Iterative Methods***

Stationary Method:
	$x_{k+1} = x_{k} + M^{-1}r_k$
**Jacobi**
  Choose M = D. (Diagnal of A).
```
% Jacobi

[~,n] = size(A);

x = zeros(n,1); r = b;

tol = 1.e-6;

for i=1:100

x = x + r./Dv; % or x = x + D \ r

r = b - A*x;

if norm(r)/norm(b) < tol, break, end

end

A*x, b % confirm Ax ≈ b
```

**Gauss-Seidel**
M = E (Lower Triangular part of A).

```
% Gauss-Seidel

[~,n] = size(A);

x = zeros(n,1); r = b;

tol = 1.e-6;

for i=1:100

x = x + E \ r;

r = b - A*x;

if norm(r)/norm(b) < tol, break, end

end

A*x, b % confirm Ax ≈ b
```

**Properties**
- Jacobi is more easily parallelized.

- Jacobi matrix M is symmetric.

- GS converges whenever Jacobi converges and often (but not always) twice as fast.

- Both Jacobi and GS converge if A is strictly (or at least irreducibly) diagonally dominant. GS is also guaranteed to converge if A is symmetric positive definite.

- Both methods are simple but converge slowly (if they converge at all). They are used as building blocks for faster, more complex methods.

- Both Jacobi and GS are simple examples of a stationary method.

**Over-relaxiation and under-Relaxiation**
Make modification on Jacobi and GS. 
$^{new}x_{k+1} = \omega x_{k+1} + (1-\omega) x_k$

When $\omega > 1$ it is an over-relaxation: 
- Base on Gauss-Seidel(GS), and $1<\omega<2$. We have **SOR**:  $x_{k+1} = x_k + \omega[(1-\omega)D + \omega E]^{-1}r_k$

When $\omega < 1$ it is an under-relaxation: 
- Base on Jacobi and $\omega = 0.8$:  $x_{k+1} = x_k + \omega D^{-1}r_k$

***Convergence of Stationary Methods***
The convergence of Stationary methods depends on the **spectral radius** of a square matrix B. It is $\rho = \max _i |\mu_i|$ (Magnitute of the largest eignvalue).

- The stationary method with iteration martrxi T will only converge iff $\rho(T) < 1$, where $T = I - M^{-1}A$
- The convergence rate is $rate = -\log_{10}\rho(T)$
- Number of iterations to reduce the error by a factor of 10 is $k = \frac 1 {rate}$
- for SOR, $\omega_{opt} = \frac 2 {1+\sqrt{1-\rho_J^2}} > 1$ where $\rho_J$ is the spectral radius of the Jacobi iteration matrix. For the model Poisson problem we have $\omega_{opt} = \frac 2 {1+ \sin(\frac \pi {N+1})}$

*Convergence of Jacobi on Poison Problem*
- O(n) Iteration is required to decrease the error by a constant factor.

So Jacobi and Gauss-Seidel requires O(n) iterations and $O(n^2)$ operations.
And SOR with optimal $\omega$ requires $O(N)$ iteraions and $O(n^{3/2})$ operations.


***Non-Stationary Method***
$x_{k+1} = x_k + \alpha_k p_k$. Where $\alpha_k$ is step size, $p_k$ is search direction.
**Gradient decent**
Take $p_k = r_k$ That is $M_k = \alpha_k^{-1}I$

To choose the step size, we use exact line search: 
- $\min_\alpha \phi(x_k+\alpha r_k) = \frac 1 2(x_k+\alpha r_k)^TA(x_k+\alpha r_k) - b^T(x_k+\alpha r_k)$

We differentiate respect to $\alpha$, and choose alpha when the diravitive = 0.

$\alpha = \frac{<r_k,r_k>} {<r_k,Ar_k>}$

We have steepest decent.

**Conjugate Gradient(CG)**
We have initial guess $x_0$ and toleranve tol. 
$r_0 = b - Ax_0$,
$\delta_0 = <r_0,r_0>$,
$b_\delta = <b,b>$,
$k = 0$,
$p_0 = r_0$.

While $\delta_k > tol^2 b_\delta$:
1. $s_k = Ap_k$
2. $\alpha_k = \frac{\delta_k}{<p_k,s_k>}$
3. $x_{k+1} = x_k + \alpha_k p_k$
4. $r_{k+1} = r_k - \alpha_k s_k$
5. $\delta_{k+1} = <r_{k+1}, r_{k+1}>$
6. $p_{k+1} = r_{k+1} + \frac{\delta{k+1}}{\delta_k}p_k$
7. k = k+1

This method is faster than Jacobi and SOR.
