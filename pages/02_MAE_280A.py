import streamlit as st
import matplotlib.pyplot as plt

st.title("MAE 280A Linear Systems Theory")

section = st.selectbox(
    "",
    [
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4",
        "Week 5"
    ],
)

if section == "Week 1":
    st.title("Week 1")
    subsection = st.radio("", ["Lecture 1", "Lecture 2", "Lecture 3"])
    
    if subsection == "Lecture 1":
        
        st.header("Lecture 1:")
        st.subheader("Thursday, September 25, 2025")
    
        st.markdown(r"""
        ### Notation
        - $R^n(C^n)$ denotes the set of n-tuples of real(complex) numbers represented as column vectors.
        - $x_i \in R$ are elements of $x$ i.e.,
        $$
        x = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} \in R^n
        $$
        - A row vector $y^T \in R^{1 \times n}$ is the transpose of the vector $y$ in $R^n$
        - For two vectors of the same length, $x, y \in R^n$:
        $$
        x^T y = \begin{bmatrix} x_1 & \dots&  x_n \end{bmatrix}\begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} = x_1 y_1 + \dots x_n y_n \in R
        $$
        
        $$
        xy^T = \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix}\begin{bmatrix} y_1 & \dots&  y_n \end{bmatrix} = 
        \begin{bmatrix} x_1 y_1 & x_1 y_2 & \dots & x_1 y_n \\ x_2 y_1 & x_2 y_2 & \dots & x_2 y_n \\ \vdots & \vdots & \ddots & \vdots \\ x_n y_1 & x_n y_2 & \dots & x_n y_n\end{bmatrix}
        $$
        
        ### Matrices
        
        - $R^{m \times n}(C^{m \times n})$ denotes the set of real (complex) $m \times n$ numbers
        - For two matrices $A \in R^{m \times n}$ and $B \in R^{m \times p} \implies AB \in R^{n\times p}$
        - $m_{ij}$: element in the $i$th row and $j$th column of $M \in R^{m \times n}$
        
        #### Defining matrix operations and special structures
        
        A matrix in $R^{n \times n}$ or in $C^{n \times n}$ (square matrices with real of complex elements) is 
        - diagonal if $a_{ij} = 0$ for $i \neq j$ e.g. $\begin{bmatrix} a_{11} & \dots & 0 \\ \vdots & \ddots & 0 \\ 0 & \dots & a_{nm}\end{bmatrix}$
        - upper triangular if $a_{ij} = 0$ for $i > j$ e.g., $\begin{bmatrix} a_{11} & \dots & \dots & a_{1n} \\ \vdots & a_{22} & \ddots & a_{2n}  \\ \vdots & \ddots & \ddots & \vdots \\ 0 & \dots & 0 & a_{nm}\end{bmatrix}$
        - lower triangular if $a_{ij} = 0$ for $i < j$ e.g., $\begin{bmatrix} a_{11} & 0 & \dots & \dots & 0 \\ a_{21} & a_{22} & 0 & \dots & 0  \\ \vdots &  & \ddots & \ddots & \vdots \\ \vdots & &  & \ddots & 0 \\ a_{n1} & \dots & \dots & \dots & a_{nm}\end{bmatrix}$
        
        - The same designations hold for block matrices: $\begin{bmatrix} A & B \\ 0 & C \end{bmatrix}$ is block upper triangular, with $A \in R^{n \times n}, B \in R^{n \times m}, C \in R^{m \times m}$
        - Toeplitz: $a_{ij} = a_{i+1, j+1}$ which looks like $\begin{bmatrix} a_{11} & a_{12} & a_{13} & a_{14} \\ a_{21} & a_{22} & a_{23} & a_{24}  \\ a_{31} & a_{32} & a_{33} & a_{34}\\ a_{41} & a_{42} & a_{43} & a_{44}\end{bmatrix} \rightarrow$ defined by first
        row and column; diagonal and all super/sub-diagonals have the same value!
        
        - $A^T$ is the transpose of $A: (A^T)_{ij} = A_{ji}$ (swap columns for rows). If $A \in R^{n \times m} \rightarrow A^T \in R^{m \times n}$. Example:
        $$
        \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6\end{bmatrix}^T = \begin{bmatrix} 1 & 3 & 5 \\2 & 4 & 6\end{bmatrix}
        $$
        - $A^H$ or $A^*$ is the Hermitian transpose or conjucate transpose of $A \in C^{m \times n}$:
        Example:
        $$
        \begin{bmatrix}1-j & 2+3j\end{bmatrix}^H = \begin{bmatrix} 1+j \\ 2- 3j\end{bmatrix}
        $$
        - a real matrix $A$ is symmetric if $A^T = A$
        - A complex matrix $A$ is Hermitian if $A^H = A$
        
        #### Multiplication of matrices and vectors
        - A matrix is a vector of vecotrs:
        $$
        U = \begin{bmatrix}u_1 & \dots & u_n\end{bmatrix} \in R^{m \times n}, u_i \in R_m\\
        V = \begin{bmatrix}v_1 & \dots & v_n\end{bmatrix} \in R^{p \times n}, v_i \in R_p\\
        \rightarrow UV^T = \begin{bmatrix}u_1 & \dots & u_n\end{bmatrix}_{m \times n}\begin{bmatrix}v_1 \\ \vdots \\ v_n\end{bmatrix}_{n \times p} = \sum_{i =1}^n u_i v_i^T \in R^{m\times p}
        $$
        
        - Let $A \in R^{n \times n}$ and $x \in R^{n \times 1}$, then:
        $$
        Ax = \begin{bmatrix}a_1 & \dots & a_n \end{bmatrix}\begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} = x_1 a_1 + \dots x_n a_n = b
        $$
        
        so that $b$ is a linear combination of columns of $A$
        Example:
        Let $A = \begin{bmatrix}1 & 2 \\3 & 4\end{bmatrix}$ and $x = \begin{bmatrix}1 \\2 \end{bmatrix}$. We define the vectors $a_1 = \begin{bmatrix}1 \\3 \end{bmatrix}$
        and $a_2 = \begin{bmatrix}2 \\4 \end{bmatrix}$
        
        Note:
        - The rows of $AB$ are linear combinations of the rows of $B$
        - The columns of $AB$ are linear combinations of the columns of $A$
        $$
        A_{m\times n}B_{n\times p} = C_{m \times p} \rightarrow \text{columns in }R^m \text{as } A\\
            \rightarrow \text{rows in }R^p \text{as }B
        $$
        
        ### Fields and Rings
        
        #### Field
        A field $(F, +, \cdot) \rightarrow$ is an object consisting of a set of elements and two binary operations
        - addition $(+)$
        - multiplication $(\cdot)$
        
        such that the followinhg axioms are obeyed for all elements of $\alpha, \beta, \gamma \in F$
        
        - Addition 
            - Associative: $(\alpha + \beta) + \gamma = \alpha + (\beta + \gamma)$
            - Commutative: $\alpha + \beta = \beta + \alpha$
            - $\exists$ identity element, we denote it by $0$ e.g., $\alpha + 0 = \alpha$
            - $\exists$ inverse: $∀ \alpha, \exists (-\alpha)$ such that $\alpha + (-\alpha) = 0$
        - Multiplication
            - Associative: $(\alpha \beta)\gamma = \alpha(\beta \gamma)$
            - Commutative: $\alpha \beta = \beta \alpha$
            - $\exists$ identity $1: \alpha \cdot 1 = \alpha$
            - $\exists$ inverse: $∀\alpha \neq 0 \exists \alpha^{-1} \in F$ such that $\alpha \cdot \alpha^{-1} = 1$
        
        Examples:
        - $(R, +, \cdot)$, $C, +, \cdot$, $(R(s), +, \cdot)$ are all fields where $R(s)$ are rational functions in $s$ with coefficients in $R$ (e.g. $\frac{s^2 = 2s + 1}{s+1}$)
        - $R[s]=$ polynomials in $s$ with coefficients in $R$ is not a field: There is no multiplicative inverse, e.g.: $s^2 + 3s$ has no inverse in $R[s](\frac{1}{s^2 + 3s}) \notin R[s]$. This leads us to the definitions of Rings
        
        #### Rings
        
        Has the same definition as a field objects, except:
        - Not necessarily commutative under multiplication
        - The inverse does not have to exist for non-zero elements under multipliction
        $Z, R[s], C[s]$ are commutative rings. $R^{n \times n}, C^{n \times n}, R^{n \times n}[s]$ are non-commutative rings
        
        
        """)
    if subsection == "Lecture 2":
        st.header("Lecture 2")
        st.subheader("Tuesday, September 30, 2025 ")
        st.markdown(r"""
        **Book**: Thomas, John B, Frank M Callier, and Charles A Desoer. Linear System Theory. Springer New York, 1991 Appendix A3
        
        ### 1. Vector Spaces
        A vector space $(V, F)$ is a set of vectors $V$ and a field of scalars $F$, and two binary operations:
        - vector addition $(+)$
        - scalar multiplication $(\cdot)$ (multiplication of vectors by scalars)
        
        such that:
        - Addition: $V \times V \rightarrow V: (x, y) \rightarrow x + y$ (closure)
            - Associative: $(x+y) + z = x + (y + z), \quad x, y, z \in V$
            - Commutative: $x + y = y + x$
            - $\exists$ identity $0$ ("zero vector"): $x + 0 = x$
            - $\exists$ inverse $∀x \in V, \quad \exists(x)$ such that $x + (-x) = 0$
            
        - Scalar Multiplication: $F \times V \rightarrow V: (\alpha, x) \rightarrow \alpha x$ (closure)
            - $(\alpha \beta)x = \alpha(\beta x)
            - $1 \cdot x = x$ where $1$ is the multiplicaiton identity of the field
            - $0 \cdot x = 0$ where $0$ is the additive identity of the field
            - $∀x \in V, ∀\alpha, \beta \in F: (\alpha + \beta)x = \alpha x + \beta x$
            - $∀x, y \in V, ∀\alpha \in F: \alpha(x + y) = \alpha x + \alpha y$
            
            Examples:
            $$
            (R^n, R)\text{yes}\quad(C^n, R)\text{yes} \quad (R^n, C) \text{no for }\alpha \in C, v\in R^n, \alpha v \in C^n \text{not }R^n
            $$
            
        #### **Exercise**: 
        Let $V = \{(a, b): a, b \in R^+\}$ where $R^+$ are positive reals. Addition is defined as $(a_1, b_1) + (a_2, b_2) = (a_1 a_2, b_1 b_2)$ 
        and multiplication: $p(a,b) = (a^p, b^p)$ for $p \in F = R$. Prove (or provide a counterexample) that $(V, R)$ with thee operations is a vector space
        
        #### Proof
        Let us prove that $V$ is a vector space with the defined operations on $R$ by showing:
        1. The addition is closed and satisfied (i), (ii), (iii), and (iv)
        2. The multiplication is closed and satisfies (v), (vi), (vii), (viii), and (ix)
        
        #### Addition

        - **Closure:**  
        Let $(a_1,b_1)$ and $(a_2,b_2) \in V$, which means $a_1,b_1,a_2,b_2 > 0$.  
        It follows directly that  
        $a_1a_2 > 0,\ b_1b_2 > 0 \implies (a_1,b_1) + (a_2,b_2) = (a_1a_2,\, b_1b_2) \in V.$

        - **Associative (i):**  
        $$
        \begin{aligned}
        [(a_1,b_1)+(a_2,b_2)] + (a_3,b_3)
            &= (a_1a_2,\, b_1b_2) + (a_3,b_3) \\
            &= (a_1a_2a_3,\, b_1b_2b_3) \\
            &= (a_1,b_1) + (a_2a_3,\, b_2b_3) \\
            &= (a_1,b_1) + [(a_2,b_2)+(a_3,b_3)].
        \end{aligned}
        $$

        - **Commutative (ii):**  
        $$
        (a_1,b_1)+(a_2,b_2)=(a_1a_2,\, b_1b_2)=(a_2a_1,\, b_2b_1)=(a_2,b_2)+(a_1,b_1)
        $$

        - **Identity (iii):**  
        $$
        (a,b)+(1,1)= (a\cdot 1,\, b\cdot 1)=(a,b)
        $$
        Here $(1,1)$ acts as the additive identity (the “zero” vector under this addition).

        - **Inverse (iv):**  
        $$
        (a,b)+\left(\frac{1}{a},\, \frac{1}{b}\right)
            = \left(a\cdot \frac{1}{a},\, b\cdot \frac{1}{b}\right)=(1,1),
        $$
        where $\frac{1}{a},\frac{1}{b}\in \mathbb{R}_+$.

        #### Multiplication (Scalar)

        - **Closure:**  
        For $p\in F=\mathbb{R}$ and $(a,b)\in V$,
        $$
        p(a,b) = (a^{p},\, b^{p}) \in V,
        $$
        since $a,b>0 \implies a^{p},b^{p}>0$.

        - **Associative (v):**  
        $$
        (pq)(a,b)=(a^{pq},\, b^{pq})=((a^{p})^{q},\, (b^{p})^{q})
        = q(a^{p},\, b^{p})=q(p(a,b)).
        $$

        - **Identity (vi):**  
        $$
        1\cdot (a,b)=(a^{1},\, b^{1})=(a,b).
        $$

        - **“Inverse” to the additive identity (vii):**  
        $$
        0\cdot (a,b)=(a^{0},\, b^{0})=(1,1),
        $$
        which is the additive identity (“zero vector”) in $(V,F,+,\cdot)$, since $(a,b)+(1,1)=(a,b)$.

        - **Distribution over scalar addition (viii):**  
        $$
        (p+q)(a,b) = (a^{p+q},\, b^{p+q})
        = (a^{p},\, b^{p}) + (a^{q},\, b^{q})
        = p(a,b)+q(a,b).
        $$

        - **Distribution over vector addition (ix):**  
        $$
        \begin{aligned}
        p[(a_1,b_1)+(a_2,b_2)]
            &= p(a_1a_2,\, b_1b_2)
            = ((a_1a_2)^{p},\, (b_1b_2)^{p}) \\
            &= (a_1^{p}a_2^{p},\, b_1^{p}b_2^{p})
            = (a_1^{p},\, b_1^{p}) + (a_2^{p},\, b_2^{p}) \\
            &= p(a_1,b_1) + p(a_2,b_2).
        \end{aligned}
        $$

        Thus, we have proved that $(V,F,+,\cdot)$ is a vector space. $\square$


        ### 2. Vector Subspaces

        Let $(V,F)$ be a vector space and $W\subseteq V$. Then $(W,F)$ is a subspace of $(V,F)$ if $(W,F)$ is itself a vector space. Moreover, $W$ is a subspace if:

        1. $W \subseteq V$ (all elements of $W$ belong to $V$).  
        2. $W$ is closed under $+$ and $\cdot$ (as defined on $V$ using $F$). This is equivalent to:  
        $$
        \forall w_1,w_2\in W,\ \forall \alpha_1,\alpha_2\in F:\ \alpha_1 w_1 + \alpha_2 w_2 \in W.
        $$

        #### **Exercise:** 
        
        Let $W_1$ and $W_2$ be subspaces of $V$.

        1. Is $W_1 \cap W_2$ a subspace?  
        2. Is $W_1 \cup W_2$ a subspace?

        ####  Solution

        1. $W_1 \cap W_2$ is a subspace.

        **Proof:**  
        (i) Proving $W_1 \cap W_2 \subseteq V$ is straightforward.  
        (ii) For closure, we want:
        $$
        \forall w_1,w_2\in W_1\cap W_2,\ \forall \alpha_1,\alpha_2\in F:\ \alpha_1 w_1 + \alpha_2 w_2 \in W_1\cap W_2.
        $$
        Take $s_1,s_2\in W_1\cap W_2$. Then:
        $$
        s_1\in W_1 \text{ and } s_1\in W_2,\quad s_2\in W_1 \text{ and } s_2\in W_2.
        $$
        Since $W_1,W_2$ are subspaces, $s_1+s_2\in W_1$ and $s_1+s_2\in W_2$, hence $s_1+s_2\in W_1\cap W_2$.  
        Similarly, for any $\alpha\in F$ and $w\in W_1\cap W_2$, we have $w\in W_1$ and $w\in W_2$, so $\alpha w\in W_1$ and $\alpha w\in W_2$, thus $\alpha w\in W_1\cap W_2$. $\square$

        2. $W_1 \cup W_2$ is **not** a subspace.

        **Proof (by contradiction):**  
        Let $s_1,s_2\in W_1\cup W_2$ with $s_1\in W_1\setminus W_2$ and $s_2\in W_2\setminus W_1$. Suppose $s_1+s_2\in W_1\cup W_2$.  
        WLOG assume $s_1+s_2\in W_1$. Since $W_1$ is a subspace, $(s_1+s_2)-s_1=s_2\in W_1$, contradicting $s_2\notin W_1$.  
        Hence $s_1+s_2\notin W_1\cup W_2$, so $W_1\cup W_2$ is not closed under addition and therefore not a subspace. $\square$

        **Proof (by counterexample):**  
        Let $V=\mathbb{R}^2$, $W_1=\text{span}\begin{bmatrix}1\\0\end{bmatrix}$, $W_2=\text{span}\begin{bmatrix}0\\1\end{bmatrix}$.  
        Then $w_1=\begin{bmatrix}1\\0\end{bmatrix}\in W_1$ and $w_2=\begin{bmatrix}0\\1\end{bmatrix}\in W_2$.  
        But $w_1+w_2=\begin{bmatrix}1\\1\end{bmatrix}\notin W_1\cup W_2$.  
        So $W_1\cup W_2$ is not closed under addition and is not a subspace. $\square$
        
        ### 3. Linear Independence and Dependence

        Suppose $(V,F)$ is a vector space.

        - A set $\{v_1,\dots,v_p\}$, with $v_i\in V$, is **linearly independent (L.I.)** iff  
        $$
        \alpha_1 v_1 + \cdots + \alpha_p v_p = 0 \implies \alpha_1=\cdots=\alpha_p=0,
        $$
        where $\alpha_i\in F$.

        - The set $\{v_1,\dots,v_p\}$ is **linearly dependent (L.D.)** iff there exist scalars $\alpha_1,\dots,\alpha_p\in F$, not all zero, such that  
        $$
        \alpha_1 v_1 + \cdots + \alpha_p v_p = 0.
        $$

        **Example:**  
        Consider the real two-dimensional vector space $(\mathbb{R}^2,\mathbb{R})$, and  
        $$
        v_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
        v_2=\begin{bmatrix}0\\3\end{bmatrix},\quad
        v_3=\begin{bmatrix}2\\6\end{bmatrix}.
        $$
        We note that  
        $$
        2v_1 + 2v_2
        = 2\begin{bmatrix}1\\0\end{bmatrix} + 2\begin{bmatrix}0\\3\end{bmatrix}
        = \begin{bmatrix}2\\6\end{bmatrix} = v_3.
        $$
        Therefore, $\{v_1,v_2,v_3\}$ is linearly dependent, since taking $\alpha_1=2$, $\alpha_2=2$, and $\alpha_3=-1$ gives  
        $$
        \alpha_1 v_1 + \alpha_2 v_2 + \alpha_3 v_3
        = 2v_1 + 2v_2 - v_3 = 0.
        $$

        **Question:** What if we consider $(\mathbb{R}^2,\mathbb{R}_+)$? Are these vectors still linearly independent?

        """)
    if subsection == "Lecture 3":
        st.header("Lecture 3")
        st.subheader("Thursday, October 2")
        st.markdown(r"""
        ### Topics
        - Bases
        - Coordinate Representation
        - Linear Maps
        
        ### Proposition: Uniqueness of Coordinates
        Let $\{b_1, b_2, \dots, b_n\}$ be a **basis** for a vector space $V$.
        Then **every** vector $x \in V$ can be written uniquely as:
        $$
        x = \alpha_1 b_1 + \alpha_2 b_2 + \dots + \alpha_n b_n
        $$
        where the scalars $\alpha_i$ are called the **coordinates** of $x$ in this basis.

        **Proof (by contradiction):**

        Suppose there exist two different coordinate representations for $x$:
        $$
        x = \alpha_1 b_1 + \alpha_2 b_2 + \dots + \alpha_n b_n \\
        x = \alpha_1' b_1 + \alpha_2' b_2 + \dots + \alpha_n' b_n
        $$

        Subtracting the two equations gives:
        $$
        0 = (\alpha_1' - \alpha_1)b_1 + (\alpha_2' - \alpha_2)b_2 + \dots + (\alpha_n' - \alpha_n)b_n
        $$

        Since $\{b_1, \dots, b_n\}$ are **linearly independent**, it must be that
        $$
        \alpha_1' - \alpha_1 = \alpha_2' - \alpha_2 = \dots = \alpha_n' - \alpha_n = 0
        $$

        Hence,
        $$
        \boxed{
        \alpha_1' = \alpha_1, \quad \alpha_2' = \alpha_2, \quad \dots, \quad \alpha_n' = \alpha_n
        }
        $$
        proving the coordinates are unique.
        
        ### Linear Maps
        Let $(V, F)$ and $(W, F)$ be vector spaces over the same field $F$.
        A function
        $$
        A: V \to W
        $$
        is called a **linear map** (or **linear transformation**) if and only if
        it satisfies the **superposition principle**:
        $$
        A(\alpha_1 v_1 + \alpha_2 v_2) = \alpha_1 A(v_1) + \alpha_2 A(v_2),
        \quad \forall v_1, v_2 \in V, \; \forall \alpha_1, \alpha_2 \in F
        $$

        ---

        ### Example
        Define a map:
        $$
        A: P_2(\mathbb{R}) \to P_2(\mathbb{R}) \\
        A(a s^2 + b s + c) = c s^2 + b s + a
        $$

        **Is $A$ linear?**

        Let
        $$
        v_1 = a_1 s^2 + b_1 s + c_1, \quad v_2 = a_2 s^2 + b_2 s + c_2
        $$
        and let $\alpha_1, \alpha_2 \in \mathbb{R}$.

        Then:
        $$
        A(\alpha_1 v_1 + \alpha_2 v_2)
        = A\big((\alpha_1 a_1 + \alpha_2 a_2)s^2 + (\alpha_1 b_1 + \alpha_2 b_2)s + (\alpha_1 c_1 + \alpha_2 c_2)\big)
        $$
        $$
        = (\alpha_1 c_1 + \alpha_2 c_2)s^2 + (\alpha_1 b_1 + \alpha_2 b_2)s + (\alpha_1 a_1 + \alpha_2 a_2)
        $$
        $$
        = \alpha_1 (c_1 s^2 + b_1 s + a_1) + \alpha_2 (c_2 s^2 + b_2 s + a_2)
        = \alpha_1 A(v_1) + \alpha_2 A(v_2)
        $$

        Therefore, $A$ is **linear**.
        
        ### Range Space and Null Space

        Let $A: U \to V$ be a **linear map** between vector spaces $U$ and $V$ over the same field $F$.

        1. **Range Space (Image)**
        $$
        R(A) := \{\,v \in V \;|\; A(u) = v \text{ for some } u \in U\,\}
        $$

        **Notes:**
        - $R(A)$ (or $\text{Im}(A)$) is a **subspace** of $V$.
        - The equation $A(u) = b$ has at least one solution **if and only if**
            $$
            b \in R(A)
            $$

        2. **Null Space (Kernel)**
        $$
        N(A) := \{\,u \in U \;|\; A(u) = 0_V\,\}
        $$
        **Note:** $N(A)$ is a **subspace** of $U$.

        ---

        ### Theorem

        Let $A: U \to V$ be linear.

        (i) The equation $A(u) = b$ has a **unique solution**  
        $$
        \iff N(A) = \{0_U\}
        $$

        (ii) Let $x_0$ be one particular solution of $A(x_0) = b$.  
        Then every solution of $A(x) = b$ satisfies
        $$
        A(x) = b \iff (x - x_0) \in N(A)
        $$

        **Proof:**

        From $A(x_0) = b$ and $A(x) = b$, subtract to get:
        $$
        A(x) - A(x_0) = 0_V \iff A(x - x_0) = 0_V
        $$
        Therefore, $(x - x_0) \in N(A)$.

        ---
        
        **(i) ⇒ direction:**  
        If $N(A) = \{0_U\}$, then for any $b \in R(A)$ there is at most one $u$ such that $A(u) = b$.  
        Because if $A(u_1) = A(u_2) = b$, then:
        $$
        A(u_1 - u_2) = 0_V \implies (u_1 - u_2) \in N(A)
        $$
        Hence, $u_1 - u_2 = 0_U \implies u_1 = u_2$.  
        Thus, the solution is unique.

        **(i) ⇐ direction:**  
        Suppose $A(u) = b$ has a unique solution for every $b \in R(A)$.  
        If there existed a nonzero $u_0 \in N(A)$ with $A(u_0) = 0_V$, then both $u_0$ and $0_U$ would map to $0_V$, contradicting uniqueness.  
        Hence, $N(A) = \{0_U\}$.

        ---

        ### Example 1: Simple Range and Null Space
        Let
        $$
        A = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}, \quad
        u = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix}, \quad
        A(u) = v = \begin{bmatrix} u_1 \\ 0 \end{bmatrix}
        $$

        Here,
        $$
        R(A) = \text{span}\!\left(\begin{bmatrix}1 \\ 0\end{bmatrix}\right), \quad
        N(A) = \text{span}\!\left(\begin{bmatrix}0 \\ 1\end{bmatrix}\right)
        $$
        confirming that $R(A) \subseteq \mathbb{R}^2$ and $N(A) \subseteq \mathbb{R}^2$ are both subspaces.

        ---

        ### Matrix Representation of a Linear Map

        Let $\{u_1, u_2, \dots, u_n\}$ be a **basis** for $U$, and
        $\{v_1, v_2, \dots, v_m\}$ be a **basis** for $V$.

        Then, the **matrix representation** of $A$ in these bases is:
        $$
        [A]_{B_V,B_U} = 
        \begin{bmatrix}
        [A(u_1)]_{B_V} & [A(u_2)]_{B_V} & \dots & [A(u_n)]_{B_V}
        \end{bmatrix}
        \in F^{m \times n}
        $$

        ---

        ### Example 2: A Map from $R^{2\times 2}$ to $P_2(\mathbb{R})$
        $$
        A: \mathbb{R}^{2\times 2} \to P_2(\mathbb{R})
        $$
        defined by
        $$
        A\!\left(\begin{bmatrix} a & b \\ c & d \end{bmatrix}\right)
        = (a - b) + (b - c)x + (c - d)x^2
        $$

        **Domain basis:**
        $$
        B_1 = \left\{
        \begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix},
        \begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix},
        \begin{bmatrix}0 & 0 \\ 1 & 0\end{bmatrix},
        \begin{bmatrix}0 & 0 \\ 0 & 1\end{bmatrix}
        \right\}, \quad \dim(B_1) = 4
        $$

        **Codomain basis:**
        $$
        B_2 = \{1, x, x^2\}, \quad \dim(B_2) = 3
        $$

        Compute $A$ on each basis element:
        $$
        \begin{aligned}
        A\!\left(\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}\right) &= 1 \\
        A\!\left(\begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}\right) &= -1 + x \\
        A\!\left(\begin{bmatrix}0 & 0 \\ 1 & 0\end{bmatrix}\right) &= -x + x^2 \\
        A\!\left(\begin{bmatrix}0 & 0 \\ 0 & 1\end{bmatrix}\right) &= -x^2
        \end{aligned}
        $$

        The **matrix representation** of $A$ in these bases is therefore:
        $$
        [A]_{B_2,B_1} =
        \begin{bmatrix}
        1 & -1 & 0 & 0 \\
        0 & 1 & -1 & 0 \\
        0 & 0 & 1 & -1
        \end{bmatrix}
        \in \mathbb{R}^{3\times4}
        $$

        ---

        ### Example 3: Computing a Transformation

        Let
        $$
        C = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}
        = 1\begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix}
        + 2\begin{bmatrix}0 & 1 \\ 0 & 0\end{bmatrix}
        + 0\begin{bmatrix}0 & 0 \\ 1 & 0\end{bmatrix}
        + 1\begin{bmatrix}0 & 0 \\ 0 & 1\end{bmatrix}
        $$
        So, in coordinates with respect to $B_1$:
        $$
        [C]_{B_1} =
        \begin{bmatrix}1 \\ 2 \\ 0 \\ 1\end{bmatrix}
        $$

        Apply $A$:
        $$
        [A(C)]_{B_2} =
        \begin{bmatrix}
        1 & -1 & 0 & 0 \\
        0 & 1 & -1 & 0 \\
        0 & 0 & 1 & -1
        \end{bmatrix}
        \begin{bmatrix} 1 \\ 2 \\ 0 \\ 1 \end{bmatrix}
        =
        \begin{bmatrix} -1 \\ 2 \\ -1 \end{bmatrix}
        $$

        Hence,
        $$
        A(C) = -1 + 2x - x^2
        $$

        ---

        ### Key Takeaways
        - $R(A)$ = image = span of columns of $[A]$
        - $N(A)$ = kernel = set of all $u$ such that $A(u) = 0$
        - If $N(A) = \{0\}$ → transformation is **injective (one-to-one)**
        - Matrix representation depends on chosen bases of domain and codomain
        """)
    
    # fig, ax = plt.subplots(figsize=(6, 4))
    
    # st.pyplot(fig)
    
if section == "Week 2":
    st.title("Week 2")
    
    w2_sub = st.radio("Lectures",
            ["Lecture 4 (raw)", "Lecture 4 (AI structured and annotated)", "Lecture 5"])
    
    if w2_sub == "Lecture 4 (raw)":
    
        st.header("Lecture 4:")
        st.subheader("Tuesday, October 7, 2025")
        st.markdown(r"""
             
        #### 1) Triangle Inequality
        $$
        ||V_1 + V_1|| = ||V_1|| + ||V_1||
        $$    
        
        #### Scalar Multiplication
        $$
        \forall \alpha \in F, \quad u \in V
        $$
        
        #### Example
        $$
        ||v|| = \sum_{i=1}^n|\alpha_1|\\
        ||x||_2 = \left(\sum_{i=1}^n |X_1|^2\right)^{\frac{1}{2}} = 2 \text{ norm}\\
        ||x||_p = \left(\sum_{i=1}^n |x_1|^p\right)^{V_p}\\
        ||x||_\infty = \text{max} |x_i| = \infty \text{ norm}
        $$
        
        #### Note:
        $$
        U + W = V\\
        \text{dim}(u + \text{dim}(w=\text{dim}(V))
        $$
        
        #### Rank:
        The rank of a matrix is the dimension of its range space.
        $$
        Rank(A) := \text{dim}(R(A))
        $$
        if a matrix is one full
        #### 1) Full row rank $\implies$ All row of A is linearly independent
        
        #### 2) Full column rank $\implies$ All columns are linearly independent
         
        ---
        ### Norms:
        
        #### Linear Spaces
        Let $(V, F)$ be vector spaces to be a norm linear space, if there $\exists$
        
        **Map**
        $$
        ||\cdot|| \to V \to R^+
        $$
        
        $$
        U + W = \left\{ \begin{bmatrix}2\\0\\0 \end{bmatrix}, \begin{bmatrix}1\\1\\0 \end{bmatrix}, \begin{bmatrix}2\\1\\0 \end{bmatrix}, \begin{bmatrix}-1\\0\\1 \end{bmatrix}\right\}
        $$
        
        $$
        \begin{bmatrix} 2 & 1 & 2 & -1 \\0& 1 & 1 & 0 \\ 0 & 0 & 0 & 1\end{bmatrix}
        $$
        
        Row reduction:
        $$
        \text{span}\left\{\begin{bmatrix}2\\0\\0 \end{bmatrix}, \begin{bmatrix}1\\1\\0 \end{bmatrix}, \begin{bmatrix}-1\\0\\1 \end{bmatrix}\right\} \in R^3
        $$
        
        ---
        ### Direct Sum
        A space $V$ is a direct sum of $U$ and $W$, denoted $\boxed{V = U \oplus W}$, if $U \cap W = \{0\}$.
        
        $V = U + W$ and ... and $U$ and $W = \{0\}$
        
        #### Example
        $$
        U = \text{span}\left\{\begin{bmatrix}1\\1\\0\end{bmatrix}, \begin{bmatrix}2\\0\\0\end{bmatrix}\right\}\\
        W = \{(x, y, z \quad x-2y + z = 0\}\\
        = \left\{(x, y, z)\begin{bmatrix}2yz\\y\\z\end{bmatrix}\right\}
        $$
        now the $F, z, y$ are free
        
        **Simplify**
        $$
        \begin{align*}
        &= \left\{(x, y, z): \begin{bmatrix}x\\y\\z\end{bmatrix}  - y\begin{bmatrix}2\\1\\0\end{bmatrix} + z\begin{bmatrix}-1\\0\\1\end{bmatrix} \quad y, z \in R \right\}\\
        &= \text{span}\left(\begin{bmatrix}2\\1\\0\end{bmatrix}, \begin{bmatrix}-1\\0\\1\end{bmatrix}\right)\\
        &\implies U + W 
        \end{align*}
        $$
        
        The range $A$ is the span of the original columns direct.
        
        **For example**
        $$
        \boxed{
        R(A) = \text{span}\left\{\begin{bmatrix}1\\0\\0\end{bmatrix}, \begin{bmatrix}-1\\1\\0\end{bmatrix}, \begin{bmatrix}0\\-1\\1\end{bmatrix}\right\}
        }
        \\
        = \text{span}(1, 1-x, -x + x^2)
        $$
        
        ---
        ### Sum of Vector Subspaces
        Let $(V, F)$ be a vector space and $u$ and $w$ be subspace of vector space $V$.
        
        Define the sum of vector subspace.
        $$
        U + W = \left\{u + w \quad u \in U, w \in W\right\} \subseteq V
        $$
        These are all elements of $u+w$.
        
        All possible combinations sums of elements $U$ and $W$
        
        $$
        \begin{align*}
        \implies N(A) = \text{span}\left(\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix} \quad Ax = 0\right)\\
        = \underbrace{\text{span}\left(\begin{bmatrix}1\\1\\1\\1\end{bmatrix}\right)}_{\text{coordinate representation in }B}
        \end{align*}
        $$
        **Check:** $A\left(\begin{bmatrix}1&1\\1&1\end{bmatrix}\right) = 0$
        
        **Range Space:**
        $$
        R(A)?
        $$
        
        **Recall:**
        $$
        R(A) :=\{b: \exists x \in u, Ax =b, b \in V\}
        $$
        
        #### Example
        The range of $A = \text{span}$
        
        Remember. This direction of the columns of A linear combinations
        
        ---
        #### What is the Null Space of $N(A)$?
        - Use the definition of the Null space of a matrix A
        $$
        N(A) := \{x \cdot Ax = 0\}
        $$
        
        Apply this to our example, what are $x$ such that
        $$
        A = \begin{bmatrix}1 & -1 & 0 & 0 \\0 & 1 & -1 & 0\\0 & 0 & 1 & -1\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix} = \begin{bmatrix}0\\0\\0\end{bmatrix}\\
        x_1 - x_4 = 0\\
        x_2 - x_4 = 0\\
        x_3 - x_4 = 0\\
        $$
        
        $$
        \implies \begin{bmatrix}1&0&0&-1\\0&1&0&-1\\0 & 1&0&-1 \end{bmatrix}\begin{bmatrix}0\\0\\0\end{bmatrix}
        $$
        
        We can easily compute the transformation of any matrix if we write in coordinate form in the basis B.
        
        $$
        C = \begin{bmatrix}1 & 2\\0&1\end{bmatrix} = 1 \cdot \begin{bmatrix}1 & 0 \\0 & 0\end{bmatrix} + 2\begin{bmatrix}0 & 1 \\0 & 0\end{bmatrix}
        + 0\begin{bmatrix}0 & 0 \\1 & 0\end{bmatrix} + 1 \begin{bmatrix}0 & 0 \\0 & 1\end{bmatrix}
        $$
        Coefficient with $B_1$
        
        $$
        A(C) = [A\cdot C] = \begin{bmatrix}1&-1&0&0\\0&1&-1&0\\0&0&1&-1\end{bmatrix}\begin{bmatrix}1\\2\\0\\1\end{bmatrix}\\
        = \begin{bmatrix}-1\\2\\-1\end{bmatrix}\\
        \implies 1 -2x - x^2
        $$
        
        $A\begin{bmatrix}1&0\\0&0\end{bmatrix}\rightarrow$ Maps to constant $1$
        $$
        \boxed{
        \left[[1]_{B_1}, [-1+x]_{B_2}, [-x+x^2]_{B_3}, [-x^2]_{B_4}\right]
        }
        $$
        $$
        \left[\underbrace{\begin{bmatrix}1\\0\\0\end{bmatrix}}_{\text{coefficient vector of }B_1}, \dots\right]
        $$
        $$
        \begin{bmatrix}
        1&-1&0&0\\
        0&1&-1&0\\
        0&0&1&-1
        \end{bmatrix} \in R^{3\times 4}
        $$
        This is a diagonal
        
        #### Consider:
        $$
        B_1 = \left\{\begin{bmatrix}1 & 0 \\0 & 0\end{bmatrix}, \begin{bmatrix}0 & 1 \\0 & 0\end{bmatrix}, \begin{bmatrix}0 & 0 \\1 & 0\end{bmatrix}, \begin{bmatrix}0 & 0 \\0 & 1\end{bmatrix}\right\}
        $$
        $$
        \boxed{\text{dim}(B_1) = 4}
        $$
        
        A basis for polynomial.
        $$
        P_2(R)\\
        B_2 = \{1, x, x^2\}\\
        \text{dim }(B_2)=3
        $$
        
        Let use compute matrix representation of A
        $$
        A = \left[ \left[A\left(\begin{bmatrix}1&0\\0&0\end{bmatrix}\right)\right]_{B_2}, \left[A\left(\begin{bmatrix}0&1\\0&0\end{bmatrix}\right)\right], \dots, \left[A\left(\begin{bmatrix}0&0\\0&1\end{bmatrix}\right)\right] \right]
        $$
        """)

    if w2_sub == "Lecture 4 (AI structured and annotated)":
        st.header("Lecture 4:")
        st.subheader("Tuesday, October 7, 2025")
        st.markdown(r"""
        #### Linear Spaces
        Let $(V, F)$ be *vector spaces*. To be a normed linear space, there $\exists$

        **Map**
        $$
        ||\cdot|| : V \to \mathbb{R}^+
        $$

        > You could expand this by defining the three axioms of a norm: positivity, homogeneity, and triangle inequality. These ensure that the mapping behaves like a proper "length" function. 

        ---

        ### Norms

        #### 1) Triangle Inequality
        $$
        ||V_1 + V_1|| = ||V_1|| + ||V_1||
        $$    

        > This appears to be a typo; likely meant $||v_1 + v_2|| \le ||v_1|| + ||v_2||$. The triangle inequality defines one of the key properties of norms. 

        #### Scalar Multiplication
        $$
        \forall \alpha \in F, \quad u \in V
        $$

        > This line seems to introduce the homogeneity property of norms: $||\alpha u|| = |\alpha|\,||u||$. 

        #### Example
        $$
        ||v|| = \sum_{i=1}^n|\alpha_i|\\
        ||x||_2 = \left(\sum_{i=1}^n |x_i|^2\right)^{\frac{1}{2}} = 2\text{-norm}\\
        ||x||_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}\\
        ||x||_\infty = \max |x_i| = \infty\text{-norm}
        $$

        > Here you could mention that these are all examples of $p$-norms. Note that $||x||_\infty = \lim_{p \to \infty} ||x||_p$. 

        ---

        ### Rank and Linear Independence

        #### Rank
        The rank of a matrix is the dimension of its range space.
        $$
        \text{Rank}(A) := \dim(R(A))
        $$

        If a matrix is *full rank*:

        #### 1) Full row rank $\implies$ all rows of $A$ are linearly independent.  
        #### 2) Full column rank $\implies$ all columns are linearly independent.

        > You could add commentary about rank-nullity theorem: $\dim(R(A)) + \dim(N(A)) = n$. 

        ---

        ### Sum of Vector Subspaces

        Let $(V, F)$ be a vector space and $U$ and $W$ be subspaces of $V$.

        Define the sum of vector subspaces:
        $$
        U + W = \left\{ u + w \;\middle|\; u \in U,\, w \in W \right\} \subseteq V
        $$

        These are all possible *sum combinations* of elements of $U$ and $W$.

        $$
        \implies N(A) = \text{span}\left(\begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix} \;\middle|\; A x = 0\right)
        = \underbrace{\text{span}\left(\begin{bmatrix}1\\1\\1\\1\end{bmatrix}\right)}_{\text{coordinate representation in }B}
        $$

        **Check:** 
        $$
        A\begin{bmatrix}1&1\\1&1\end{bmatrix} = 0
        $$

        > Consider adding a brief explanation: The sum $U+W$ is the smallest subspace containing both $U$ and $W$. 

        ---

        ### Direct Sum

        A space $V$ is a direct sum of $U$ and $W$, denoted $\boxed{V = U \oplus W}$, if $U \cap W = \{0\}$.

        Thus, $V = U + W$ and $U \cap W = \{0\}$.

        #### Example
        $$
        U = \text{span}\left\{\begin{bmatrix}1\\1\\0\end{bmatrix}, \begin{bmatrix}2\\0\\0\end{bmatrix}\right\}\\
        W = \{(x, y, z) \;|\; x - 2y + z = 0\} = \left\{(x, y, z) = y\begin{bmatrix}2\\1\\0\end{bmatrix} + z\begin{bmatrix}-1\\0\\1\end{bmatrix}\right\}
        $$

        > This could use elaboration on how the $W$ basis is derived from solving the constraint $x - 2y + z = 0$. 

        Simplify:
        $$
        \begin{align*}
        W &= \left\{(x, y, z): \begin{bmatrix}x\\y\\z\end{bmatrix} = y\begin{bmatrix}2\\1\\0\end{bmatrix} + z\begin{bmatrix}-1\\0\\1\end{bmatrix}, \; y, z \in \mathbb{R} \right\}\\
        &= \text{span}\left(\begin{bmatrix}2\\1\\0\end{bmatrix}, \begin{bmatrix}-1\\0\\1\end{bmatrix}\right)\\
        &\implies U + W = \mathbb{R}^3
        \end{align*}
        $$

        > Add a note that since $U$ and $W$ span $\mathbb{R}^3$ and have trivial intersection, $V = \mathbb{R}^3 = U \oplus W$. 

        ---

        ### Range and Null Space Relationship

        The range of $A$ is the span of its column vectors.

        **Example**
        $$
        \boxed{
        R(A) = \text{span}\left\{
        \begin{bmatrix}1\\0\\0\end{bmatrix},
        \begin{bmatrix}-1\\1\\0\end{bmatrix},
        \begin{bmatrix}0\\-1\\1\end{bmatrix}
        \right\}
        }
        $$

        > It might help to note that $R(A)$ is also called the column space of $A$, and its dimension is the rank. 

        ---

        ### Null Space Example

        **Definition:**
        $$
        N(A) := \{x \;|\; A x = 0\}
        $$

        Apply to example:
        $$
        A = 
        \begin{bmatrix}
        1 & -1 & 0 & 0 \\
        0 & 1 & -1 & 0 \\
        0 & 0 & 1 & -1
        \end{bmatrix},
        \quad
        x = \begin{bmatrix}x_1\\x_2\\x_3\\x_4\end{bmatrix}
        $$

        Compute:
        $$
        A x = 0 \implies
        \begin{cases}
        x_1 - x_2 = 0 \\
        x_2 - x_3 = 0 \\
        x_3 - x_4 = 0
        \end{cases}
        \Rightarrow x_1 = x_2 = x_3 = x_4
        $$

        Thus:
        $$
        N(A) = \text{span}\left(\begin{bmatrix}1\\1\\1\\1\end{bmatrix}\right)
        $$

        > You could note that this means $\dim(N(A)) = 1$ and $\dim(R(A)) = 3$, satisfying the rank-nullity theorem for $A \in \mathbb{R}^{3\times4}$. 

        ---

        ### Basis Representations and Coordinate Transformations

        We can compute transformations of any matrix by writing it in coordinate form with respect to a basis $B$.

        $$
        C = \begin{bmatrix}1 & 2\\0 & 1\end{bmatrix}
        = 1 \cdot \begin{bmatrix}1 & 0 \\0 & 0\end{bmatrix}
        + 2\begin{bmatrix}0 & 1 \\0 & 0\end{bmatrix}
        + 0\begin{bmatrix}0 & 0 \\1 & 0\end{bmatrix}
        + 1 \begin{bmatrix}0 & 0 \\0 & 1\end{bmatrix}
        $$

        Coefficient with $B_1$
        $$
        A(C) = [A \cdot C] = 
        \begin{bmatrix}1&-1&0&0\\0&1&-1&0\\0&0&1&-1\end{bmatrix}
        \begin{bmatrix}1\\2\\0\\1\end{bmatrix}
        = \begin{bmatrix}-1\\2\\-1\end{bmatrix}
        $$

        $$
        \boxed{1 - 2x - x^2}
        $$

        > Here, it would be helpful to describe what transformation $A$ represents (e.g., discrete differentiation on polynomial coefficients). 

        ---

        ### Polynomial Basis Examples

        $$
        A\begin{bmatrix}1&0\\0&0\end{bmatrix} \rightarrow \text{maps to constant } 1
        $$

        $$
        \boxed{
        \left[[1]_{B_1}, [-1+x]_{B_2}, [-x+x^2]_{B_3}, [-x^2]_{B_4}\right]
        }
        $$

        $$
        \left[\underbrace{\begin{bmatrix}1\\0\\0\end{bmatrix}}_{\text{coefficient vector of }B_1}, \dots\right]
        $$

        $$
        \begin{bmatrix}
        1&-1&0&0\\
        0&1&-1&0\\
        0&0&1&-1
        \end{bmatrix} \in \mathbb{R}^{3\times4}
        $$

        This is *a diagonal-like difference operator.*

        #### Consider:
        $$
        B_1 = \left\{
        \begin{bmatrix}1 & 0 \\0 & 0\end{bmatrix},
        \begin{bmatrix}0 & 1 \\0 & 0\end{bmatrix},
        \begin{bmatrix}0 & 0 \\1 & 0\end{bmatrix},
        \begin{bmatrix}0 & 0 \\0 & 1\end{bmatrix}
        \right\}
        $$
        $$
        \boxed{\dim(B_1) = 4}
        $$

        A basis for polynomials:
        $$
        P_2(\mathbb{R}), \quad
        B_2 = \{1, x, x^2\}, \quad
        \dim(B_2) = 3
        $$

        Let us compute the matrix representation of $A$:
        $$
        A = \left[
        \left[A\left(\begin{bmatrix}1&0\\0&0\end{bmatrix}\right)\right]_{B_2},
        \left[A\left(\begin{bmatrix}0&1\\0&0\end{bmatrix}\right)\right],
        \dots,
        \left[A\left(\begin{bmatrix}0&0\\0&1\end{bmatrix}\right)\right]
        \right]
        $$

        > This could be expanded to explain how $A$ acts as a linear transformation between basis spaces $B_1$ and $B_2$. 
        """)

    if w2_sub == "Lecture 5":
        st.header("Lecture 5: Normed and Inner Product Spaces")
        st.subheader("Thursday, October 9, 2025")
        st.markdown(r"""
        $(V, F)$ is a normed linear space if the units are:
        $$
        ||\cdot|| : V \rightarrow R^+
        $$ 
        
        and satisfies:
        #### 1) $||v_1 + v_2|| \leq ||v_1|| + ||v_2||$
        
        #### 2) $||\alpha V|| = |\alpha|||v||, \quad \alpha \in R, C$
        
        #### 3) $||v|| = 0 \iff v = 0_v$
        
        ---
        ### Example 1) Vector spaces of matrices, $A\in F^{n\times n}, F \in R, C$
        $a)$ $|A|_1 = \sum_{i=1}^n \sum_{j=1}^n |a_{ij}|$
        
        $b) |A|_F = (\sum_{i=1}^n \sum_{j=1}^n |a_{ij}|^2)^{1/2}$
        
        $c) |A|_b = \text{max}_{ij} \{|a_{ij}|\}$
        > Note: induced norm will be discussed later
        
        ---
        ### Example 2) Vector spaces of functions
        Let $(V, F)$ be a linear vector space. let $D$ be a set. 
        
        Let $M: D\to V$ be a class of functions. On $M$, we define addition and scalar multiplication as:
        $$
        (f+g)(d) = f(d) + g(d), \quad \forall f, g \in M, \forall d \in D\\
        (\alpha f)(d) = \alpha \cdot f(d), \quad \forall f \in M, \forall \alpha \in F, \forall d \in D
        $$
        
        ---
        ### Example 3) Continuous functions in $[t_0, t_1] \to R^n$ denoed as:
        $$
        C\left([t_0, t], R^n\right)
        $$
        
        ### Example 4) norms on vector spaces of functions
        $a) \,  ||f||_1 = \int_{t_0}^t ||f(t)||dt \rightarrow L1 \text{ norm }, \quad \text{where } ||\cdot|| \text{{ is any norm}}$
        
        $b) \, ||f||_2 = \left(\int_{t_0}^t ||f(t)||^2 dt \right)^{1/2} \rightarrow L2 \text{ norm}$ 
        
        $c) \, ||f||_\infty = \text{max}_{t\in [t_0, t_1]}\{||f(t)||\} \rightarrow L\infty \text{ norm}$
        
        ---
        ### Equivalence of norms: Two norms $||\cdot||_a$ and $||\cdot||_b$ are equivalent if 
        
        $$
        \exists\, \alpha, \beta \in \mathbb{R}^+ \text{ such that } 
        \forall v \in V,\; 
        \alpha\, ||v||_a \leq ||v|| \leq \beta\, ||v||_b
        $$
        
        ---
        ### Example 5) Consider the vector space $(F^n, F)$. Then
        
        #### 1) $||x||_\infty \leq ||x||_1 \leq n||x||_\infty$
        Proof: Definition of the 1-norm
        
        $$
        ||x||_1 = \sum_{i=1}^n |x_i| = \underbrace{\text{max}_{i = 1, \dots, n}\{|x_i|\}}_{||x_\infty||} + \sum_{i = 1, \dots, n; 1 \neq \text{ max}}|x_i|\\
            \downarrow \\
        ||x||_\infty \leq ||x||_\infty + \sum_{i = 1, \dots, n; i \neq \text{ max}}|x_i| \leq n \cdot ||x||_\infty
        $$
        
        ---
        #### 2) $||x||_\infty \leq ||x||_2 \leq \sqrt{n}||x||_\infty$
        $$
        \begin{align*}
        ||x||_2 = \left(||x||_\infty^2 + \sum_{i = 1; i \neq \text{ max}}|x_i|^2\right)^{1/2} &\leq \left(||x||_\infty^2 + \sum_{i=1; i \neq \text{ max}}(\text{max}_{j=1\dots n}\{|x_j|\}^2)\right)^{1/2}\\
        &= \left(||x||_\infty^2 + (n-1)(\text{max}_{j=1\dots n}\{|x_j|\}^2)\right)^{1/2}\\
        &= (n \cdot ||x||_\infty^2)^{1/2}\\
        &= \sqrt{n} \cdot ||x||_\infty
        \end{align*}
        $$
        
        So, $\boxed{||x||_2 \leq \sqrt{n} \cdot ||x||_\infty}$
        
        ---
        #### 3) $\frac{1}{\sqrt{n}}||x||_1 \leq ||x||_2 \leq ||x||_2$
        $$
        \frac{1}{\sqrt{n}}||x||_1 \leq ||x_2|| \leq ||x||_1
        $$        
        
        **By definition:** $||x||_1 = \sum_{i=1}^n|x_i| \quad ||x||_2 = \left(\sum_{i=1}^n|x_i|^2\right)^{1/2}$
        $$
        ||x||_2 = \left(\sum_{i=1}^n |x_i|^2\right)^{1/2} \leq \left(\left(\sum_{i=1}^n|x_i|\right)^2 \right)^{1/2} \rightarrow \text{ Triangle Equality}\\
        = \sum_{i=1}^n|x_i|=||x||_1
        $$
        
        Thus: $\boxed{||x||_2 \leq ||x||_1}$
        
        $$
        ||x||_1 = \sum_{i=1}^n|x_i| \iff ||x||_1^2 = \left(\sum_{i=1}^n|x_i|\right)^2
        $$
        $$
        \frac{1}{n} ||x||_1^2 = \frac{1}{n}\left(\sum_{i=1}^n|x_i|\right)^2 \geq \frac{1}{n}\left(\sum_{i=1}^n|x_i|^2\right)
        $$
        
        ---
        ### Hilbert Spaces
        Let $F = R$ or $F = C$ and consider the linear space $(H, F)$. The function $<\cdot, \cdot>: H \times H \to F$ is called an inner product if and only if:
        
        $a) \, <x, y+z> = <x, y> + <x, z>, \quad \forall x, y, z \in H$
        
        $b) \, <x, \alpha y> = \alpha <x, y>, \quad \forall x, y \in H, \forall \alpha \in F$
        
        $c) \, <x, x, > > 0 \iff x\neq 0$
        
        $d) \, <x, y> = <y, x> \implies \text{ complex conjugate}$
        
        A vector space equipped with an inner product is called a Hilbert space
        
        ---
        ### Example 6) $(F^2, F, <\cdot, \cdot>)$, with $F = R$ or $F = C$ is a Hilbert space under the inner product:
        $$
        <x, y> := \sum_{i=1}^n y_i \bar{x_i} = x^H u
        $$
        
        > $H$ is Hermitian transpose or $\dagger$

        ---
        ### Orthogonality: COnsider the Hilbert Space $(H, F, <\cdot, \cdot>)$. Two vectors are orthogonal if and only if
        $$
        <x, y> = 0
        $$
        
        ---
        ### Example 7) Consider $R^2$
        - $v_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $v_2 = \begin{bmatrix} 0 \\ 1\end{bmatrix}$ are orthogonal
        $$
        <v_1, v_2> = v_1^T v_2 = 0
        $$
        - $w_1 = \begin{bmatrix}-1\\1\end{bmatrix}$ and $w_2 = \begin{bmatrix}-1\\1\end{bmatrix}$
        $$
        <w_1, w_2> = w_1^Tw_2 = 0 \rightarrow w_1 \perp w_2
        $$
        
        ---
        ### Orthogonal Complement of a Subspace:
        Let $M \subseteq H$; the subset $M^\perp := \{y\in H: <x, y> = 0, \forall x\in M\}$
        
        ---
        ### Example 8) Let $M = \text{span}\{v_1, v_2\}, v_1 = \begin{bmatrix}-1\\1\\0\end{bmatrix}, v_2 = \begin{bmatrix}0\\0\\1\end{bmatrix}.$ Find $M^{\perp}$
        
        > Note: $M\subset R^3$, and $dim(M) = 2$
        
        **Proof:** From the definition of $M^\perp$, we get
        $$
        \begin{align*}
        M^\perp &= \{w: v_1^T w = 0 \text{ and } v_2^Tw = 0, w\in R^3\}\\
        &= \left\{w: \begin{bmatrix}v_1^Tw \\v_2^Tw\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}, w \in R^3 \right\}\\
        &= \left\{w: \begin{bmatrix}v_1^T \\v_2^T\end{bmatrix}w = \begin{bmatrix}0\\0\end{bmatrix}, w \in R^3 \right\} = N\left(\begin{bmatrix}v_1^T\\v_2^T\end{bmatrix}\right)
        \end{align*}
        $$
        Now find the null space of the matrix:
        $$
        \begin{bmatrix} v_1^T \\v_2^T\end{bmatrix} = \begin{bmatrix}-1&1&0\\0&0&1\end{bmatrix} \implies \begin{bmatrix}-1&1&0\\0&0&1\end{bmatrix}\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix} = \begin{bmatrix}0\\0\end{bmatrix}\\
        \implies x_1 = x_2, x_3 = 0 \implies w = \begin{bmatrix}1\\1\\0\end{bmatrix}
        $$
        Thus: $M^\perp = \text{span}\left\{\begin{bmatrix}1\\1\\0\end{bmatrix}\right\} \rightarrow R^3 = \text{span}\left\{\begin{bmatrix}-1\\1\\0\end{bmatrix}, \begin{bmatrix}0\\0\\1\end{bmatrix}\right\} \oplus \text{span}\left\{\begin{bmatrix}1\\1\\0\end{bmatrix}\right\}$
        
        ---
        ### Fundamental Theorem of Linear Algebra:
        
        #### Theorem: Let $A:$R^n \to R^m$, then:
        
        $(i) \, N(A)^\perp = R(A^T)$
        
        $(ii) \, R(A)^\perp = N(A^T)$
        
        both "$=$" is $\subseteq$ AND $\supseteq$
        
        #### Proof of $(i)$: We note that showing $(i)$ is equivalent to showing 
        $$
        N(A) = R(A^T)^\perp
        $$
        
        #### "$\subseteq$": Let $x \in R^n, x\in N(A)$.
        
        Then $iff Ax = 0 \implies \underbrace{y^T A}_{(A^Ty)^T}x = y^T 0 = , \forall y \in R^m$
        $$
        \implies x \perp (A^Ty)\\
        \implies x \perp R(A^T)\\
        \implies \text{some } x \in N(A), \text{ we have } N(A) \subseteq R(A^T)^\perp
        $$
        
        #### "$\supseteq$": Let $x \in R(A^T)^\perp$
        $$
        \begin{align*}
        \text{Then } x \in R(A^T)^\perp &\iff (A^Ty)^Tx = 0, \forall y \in R^m\\
        &\iff y^T(Ax) = 0, \forall y \in R^m\\
        &\implies Ax = 0 \iff x \in N(A)
        &\iff R(A^T)^\perp \subseteq N(A)\\
        &\implies N(A)^\perp = R(A^T)
        \end{align*}
        $$
        """)

if section == "Week 3":
    st.title("Week 3")
    
    st.header("Lecture 6:")
    st.subheader("Tuesday, October 14, 2025")
    st.markdown(r"""
    
    #### **Following from Lecture 5** 
    
    **Proposition 1**:
    
    Let A $\in R^{n \times n}$ , Any $v \in R^n$ can be written uniquely as $v = x + y$, where $x \in N(A)$
    and $y = N(A)^{\perp} = R(A^T)$
    
    $$
    \boxed{
    \implies R^n = N(A) \oplus R(A^T)
    }
    $$
    
    **Proposition 2:**
    Consider $A\in R^{m \times n}$. Any $w \in R^m$ can be uniquely written as $w = x = y$, where $x \in R(A)$ 
    $y \in R(A)^{\perp} = N(A^T)$
    
    $$
    \boxed{
    \implies R^m = R(A) \oplus N(A^T)
    }
    $$
    
    #### Functions and Maps
    
    Let $A: V \rightarrow W, A(v) = w.$ 
    
    Then,
    
    (i) $A$ is surjective/onto if $R(A) = W$
    - Note: if $A(v) = Av$, then the map is surjective if $\text{rank}(A) = m$ (rows are LI)
    
    (ii) A is injective 1-1 if $N(A) = \{0\}$
    - Note: If $A(v) = Av$, then the map is 101 if $\text{rank}(A) = m$ (cols are LI)
    - The map $A(\cdot)$ is bijective (injective + surjective) wher the matrix is invertible
    
    #### Determinants
    Let $A = \begin{bmatrix} a & b \\ c & d\end{bmatrix} \quad a, b, c, d \in F$
    
    Then:
    
    $$
    |A| := \text{determinant of } A = ab - bc
    $$
    
    For $A \in R^{n \times n}, n > 2$, the determinant is defined by induction.
    
    $$
    |A| = (-1)^{i+1}a_{i1}|A_{i1}| + \dots + (-1)^{i+n}a_{in}|A_{in}| \\
    = \sum_{i=1}^n (-1)^{i+j}a_{ij}\text{ det }|A_{ij}| 
    $$
    
    Example:
    
    Let $A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33}\end{bmatrix}, \quad \text{conjugate } A$
    
    Solution:
    
    $$
    |A| = \sum_{i=1}^n (-1)^{i+j}a_{ij}\text{ det }|A_{ij}| \\
    = (-1)^{i+1}a_{11} \begin{bmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{bmatrix} \dots \\
    $$
    
    #### Properties of Determinants
    1) $det(A) = det(A^T)$, $det(A) = det(A^\dagger)$
    2) A zero row or column of A means $det(A) = 0$
    3) Interchanging a row or a column of $A$ changes the sign of the determinant
    4) Multiplying a row of $A$ by a constant $\alpha$ multiplies $det(A)$ by $\alpha$
    $$
    \implies |\alpha A| = \alpha^n |A|
    $$
    5) Multiplying a row by a scalar and adding it to another row does not change the determinant
    6) If I have two matrices $A, B \in R^{m \times n}$ then $|A||B| = |AB| \implies |A^{-1}| = \frac{1}{|A|}$ only if $A$ is invertible
    7) If $A$ is diagonal $\implies |A| = \prod_{i=1}^n a_{ii}$ also lower triangular.
    8) $A$ is invertible $\iff \text{rank}(A) = n$ $\iff |A| = \text{not zero} \iff N(A) = \{0\}$
    9) $A$ is invertible $\iff (A^T) = (A^{-1})^T$
    
    #### Eigenvalues and Eigenvectors
    $\lambda \in C$ is an eigenvalue of $A\in R^{n \times n}$ and $u \in C^n$ is the coresponding eigenvector if:
    $$
    Au = \lambda u, \quad u \neq = 0\\
    \iff (A-\lambda I)u = 0\\
    \iff u \in N(A - \lambda I)
    $$
    
    For any eigenvalue $\lambda$ the characteristic polynomial is zero.
    $$
    \gamma(\lambda) = det(A - \lambda I) = 0
    $$
    
    #### Definition
    A is diagonalizable if it has $n$ linearly indipenent eigenvectors $u_1, \dots u_n$
    $$
    \implies Au := \lambda_i u_i \\
    \implies AU = U \Lambda 
    \iff A = U \Lambda U^{-1}, \quad U^{-1}
    $$
    exists because of linear independence
    
    #### Spectral Theorem
    Let $A \in R^{n \times n}$, $A = A^T$ and let $\lambda_i = R, i = 1, \dots n$
    be the eigenvvalues that exists $u_i \in R^n$ $u_i \perp u_j$, $i \neq j$
    $||u_i|| i$ (e.g. $u_i, \dots u_n$ is orthonormal)
    $$
    A = U \Lambda U^{-1}
    $$
    since $u_i$ are orthonomal, $u^T u = I = U U^T$
    
    Every real symmetric matrix has an orthonormal set of eigvenvectors
    
    #### Positive Definite Matrices
    Let $A \in S^{n \times n}$, A is positive semidefinite $A \geq 0$ 
    
    If 
    $$
    x^T Ax > 0, \quad x \in R^n \neq 0
    $$
    then $A$ is called positive definite
    
    **Note**:
    1) $A \geq 0 \iff \lambda_i(A) \geq 0, i = 1, \dots, n$
    2) $A > 0 \iff \lambda_i(A) > 0$
    3) For any $A \in R^{n \times m}$
        - $A^T A \geq 0$ and $AA^T \geq 0$
        - $AA^T \iff A$ is full column rank ($\text{rank}(A) = n$)
        - $AA^T$ > 0 \iff A is full row rank m
    $$
    x^TAx = (x^T U \Lambda U^{-1}x)
    $$
    """)

    st.divider()
    st.header("Lecture 7:")
    st.subheader("Thursday, October 16, 2025")
    st.markdown(r"""
       
    #### Singular Value Decomposition
    
    Any non-zero matrix $A \in R^{n \times m}$ an be decomposed as 
    $$
    A = U S V^T
    $$
    
    where 
    $$
    U^T U = UU^ = I_m, V^TV = VV^T = I_n, S = \text{diag}(\sigma_1, \dots, \sigma_i)
    $$                
    where $\sigma$ are positive definite and $v = \text{rank}(A)$
    
    The first columns of $U$ are the left singular vector
    $$
    A = UU^{-1} \\
    Ax = \lambda x \\
    Av_i = \sigma_i u_i \\
    A^T u_i = \sigma_i v_i
    $$
    
    Note:
    1) $A^T A = (USV^T)^T(USV^T) = VSU^TUSV^T = VS^2V^T \in R^{n \times n}$
    2) $AA^T = USV^T(USV)^T = USV^TVSU = US^2U \in R^{m \times m}$
    
    The non-zero singular values of $A$ are the square roots of the eigenvalues of $A^T A$ or $AA^T$
    
    #### Theorems
    **$7.1$**: If $A = A^\dagger$, the Eigenvalue Decomposition and Singular Value Decomposition are related
    $$
    | \lambda_i (A)|  = \sigma_i (A) 
    $$
    
    **$7.2$** For $A \in R^{m\times m} |\text{det}(A)| = \prod_{i=1}^m \sigma_i(A)$
    
    - Proof: $|det(A)| = |det(U)||det(S)||det(V)| = |det(S)| = \prod_{i = 1}^m \sigma_i (A)$
    - $|det(U)| = |det(V)| = 1$
    
    #### Properties
    
    Why is $det(U) = 1$ if $U^T U = I$?
    $$
    |A \cdot B| = |A||B|
    $$
    $$
    1 = |I| = |U^T U| = |U^T||U| = |U|^2
    $$
    
    #### Induced Norms
    Let $A: = U \rightarrow V$ be a linear map, where on $U$ we have the norm $|||_u$ and on $V$ we have $||||_v$
    
    The induced matrix norms are defined as
    $$
    ||A||_p := \text{sup}_{x \neq 0}\frac{||Ax||_p}{||\lambda||_p} = \text{sup}_{||\lambda||_{p=1}} ||Ax||_p
    $$
    - supremum means maximum over all $x$ which requires to search the whole space
    
    Example: 
    $$
    ||A|_1 = \text{sup}_{x\neq 0} \frac{||Ax||_1}{||\lambda||_1} \quad ||A|_2 = \text{sup}_{x\neq 0} \frac{||Ax||_2}{||\lambda||_2}
    $$
    
    Note: Computing norms through this definition requires $\infty$ many vectors so not feasible!
    
    #### Relation of SVD to matrix norms
    **$7.3$**: 
    $$
    ||A||_F^2 = \sum_{i = 1} \sigma_i^2 = \text{trace}(A^\dagger A) = \text{sum of all diagonal indices of a matrix}
    $$
    - Not individual norm. Frobenius norm
    $$
    ||A||_2 = \sigma_i (A) = \text{max singular value of } A
    $$
    
    **Proof**
    $$
    \text{sup}||Ax||_2 = \text{sup}\sqrt{x^T A^T A x} \\
    = \text{sup}\sqrt{x^T VSU^TUSV^T x} \\
    = \text{sup}\sqrt{x^T VS^2V^T x} \\  
    \text{Let }: V^Tx = y \\
    = \text{sup}\sqrt{y^T S^2 y} \\ 
    = \text{sup}\sqrt{\sigma_1^2 y_1^2 + \dots + \sigma_i^2 y_i^2} \\ 
    = \sqrt(\sigma_1^2 1^2) = \sigma(A)
    $$
    
    #### Diferential Equations (C + D Appendix B)
    Let $x(t) \in R^n$ be the system state (internal description) $t \geq 0$ time, and let $f(\dot) = R^n \rightarrow R$ (nonlinear drift)
    
    Then:
    $$
    \dot{x}(t) = f(x, t), x(0) = x_0 \quad (IVP)
    $$
    
    Is a vector valued autonomous ODE
    
    #### Questions
    1) Is there a solution to the IVP (existence) ?
    2) Are there many solutions (uniqueness) ?
    
    
    #### Definitions
    1) A function $f(\cdot)$ is continuous if for every $\epsilon > 0$ there exists $\delta > 0 $ such that 
    if $|| x_1 - x_2 || < \delta \implies ||f(x_1) - f(x_2)|| < \epsilon$ 
    2) $f(x, t)$ is piecewise continuous in $t$ for all $x$ if $f(x)$ goes from $R_f \rightarrow R_n$ os continuous except for finitely many points of discontinuity in any compact time interval (bounded and closed)
        - Closed and Bounded: $f(x)$ is continuous except at a finite number of points
    3) $f(x, t): R_t \rightarrow R^n$ is Lipschitz continuous (LC) in $x$ for all $t\in R_t$ if there exists a piecewise continuous
    function $k(\cdot): R_t \rightarrow R_\perp$ such that
    $$
    ||f(x, t) - f(y, t)|| \leq k(t) ||x - y|| \quad \text{for all } x, y \in R
    $$
    - Note: Lischitz continuous $\implies$ continuous but not in reverseå
    
    #### Fundamental Theorem of Differential Equations
    For $x(t) \in R^n$, $f(x, t): R_t \rightarrow R_n$ consider the IVP
    $$
    \dot{x}(t) = f(x, t) \\
    x(0) = x_0
    $$
    If $f(x, t)$ is piecewise continuous in $t$ for all $x\in G \leq R^n$ 
    and if $f(x, t)$ is Lipschitz continuous in x for all $t\in [t_0, t_1]$
    
   then, there exists a unique solution $\Phi(\cdot): R_t \rightarrow R^n$ to the IVP which is continuously differentiable almost 
   everwhere satisfies IVP for all $t \in [t_0, t_1]$ where $D$ is the set of discontinuity points for $f(x, t)$ in $t$
   
   
    """)
    
if section  == "Week 4":
    st.title("Week 4")
    st.header("Lecture 8:")
    st.subheader("Tuesday, October 21, 2025")
    st.markdown(r"""
    #### Recap
    
    Non-autonomous:
    $$
    \dot{x}(t) = f(x(t), t) = f(x) + Bu(t)    
    $$
    
    Autonomous:
    $$
    \dot{x}(t) = f(x(t) = f(x))
    $$
    
    ---
    ### Bellman-Gromwell Lemma (Integral Form)
    Let $u(\cdot) > 0, k(\cdot) > 0$ real-valued  pointwise continuous on $R_t$ and assume $c_1 \geq 0$, and $t_0 \in R_t$
    
    If:
    $$
    u(t) \leq c_1 + \int_{t_0}^t k(\tau)u(\tau)d\tau
    $$
    
    Then
    $$
    u(t) \leq c_1 \text{exp}\left(\int_{t_0}^t k(\tau)d\tau\right)
    $$
    
    Proof: Without loss of generality (WLOG) assum $t > t_0$ and let
    $$
    U(t) = c_1 + \int_{t_0}^t k(\tau)u(\tau)d\tau
    $$
    $$
    {\text{by def}} \implies u(t) \geq U(t) \quad (3)
    $$
    Multiply $(3)$ by $k(t)\text{exp}\left( -\int_{t_0}^t l(\tau)d\tau) \right)$
    $$
    \rightarrow u(t) k(t) \text{exp}\left( -\int_{t_0}^t k(\tau)d\tau) \right) - U(t) k(t)\text{exp}\left( -\int_{t_0}^t k(\tau)d\tau \right) \leq 0\\
    \text{use} \frac{d}{dx}\left(\int_0^x f(t) dt\right) = f(x)\\
        \downarrow \\
    \implies \frac{d}{dt}\left (U(t) k(t)\text{exp}\left( -\int_{t_0}^t k(\tau)d\tau \right) - c_1\leq 0 \right)
    $$
    
    Integrating in $[t_0, t]$
    
    $$
    U(t) k(t)\text{exp}\left( -\int_{t_0}^t k(\tau)d\tau \right) - c_1\leq 0 \\
    u(t) \leq U(t) \leq c_1 \text{exp}\left(\int_{t_0}^t k(\tau)d\tau \right) \square
    $$
    
    Aside (by chain rule):
    $$
    \frac{d}{dt}\left(U(t)\text{exp}\left(-\int_{t_0}^t k(\tau)d\tau \right)\right)\\
        =U'(t)\text{exp}\left(-\int_{t_0}^t k(\tau)d\tau \right) + U(t)\text{exp}(k(t))\left(-\int_{t_0}^t k(\tau)d\tau \right) \leq 0
    $$
    
    ---
    #### Uniqueness of Solutions
    Consider
    $$
    \dot{x} = f(x(t), t), \quad x(t_0) = x_0, \quad t_0 \in R_t \quad (4)
    $$
    
    Part of the assertion of the Fundamental Theorem of Differential Equations, if $(f\cdot)$ is 
    continouous and Lipschitz continuous in $x$ and we have a unique solution.
    
    **Proof of Uniqueness**:$\\$
    Asssume we showed existance already. Lets show the solution is unique, we prove by contradiction.
    
    > ""We take the difference of them and want that difference to be zero""
    
    Assume $\exists$, two solutions to $(4)$, call them $\phi(\cdot)$ and $\psi(\cdot)$
    
    $$
    \dot{\phi(t)} = f(\phi(t), t), \quad \phi(t_0) = x_0 \quad (5)\\
    \dot{\psi(t)} = f(\psi(t), t), \quad \psi(t_0) = x_0 \quad (6)\\
    \implies \int (5) - \int (6) \implies \phi(t) - \psi(t) = \int_{t_0}^t f(\phi(\tau), t)d\tau - \int_{t_0}^t f(\psi(\tau), t)d\tau\\
    |||\phi(t) - \psi(t)|| \leq \int_{t_0}^t f(\phi(\tau), t)d\tau - \int_{t_0}^t f(\psi(\tau), t)d\tau
    $$
    
    Since $f(\cdot)$ is Lipschitz continuous on $x \implies \exists k(t)$ such that
    $$
    \leq \int_{t_0}^t k(\tau)|||\phi(\tau) - \psi(\tau)|| d\tau
    $$
    
    Let $k = \text{max}k(t)$. Then $[t_0, t]$
    $$
    ||\phi(t) - \psi(t)|| \leq 0 + \int_{t_0}^t k ||\phi(\tau) - \psi(\tau)||d\tau \\
    \implies ||\phi(t) - \psi(t)|| \leq c_1 \text{exp}\left(-\int_{t_0}^t k dt \right) = 0, \quad c_1 = 0\\
    \implies \phi(t) = \psi(t)
    $$
    
    ---
    #### Dynamical Systems (Callier + Desson, Ch. 5)
    
    We denote a dynamical system formally as 
    $$
    D = (U, \sum, y, S, \gamma)
    $$
    and $\tau$ is $\tau = t \in R_t, \tau = kT, k \in Z$
    
    whereß
    - $U :=$ set of input functions $u(t): T \rightarrow U = R^n$
    - $Y :=$ set of output functions, $y(t): T \rightarrow Y = R^n$
    - $\sum :=$ state space $\quad x(\cdot): T \rightarrow Y = R^n$
    - $S :=$ state transition function $S(\cdot, \cdot, \cdot, \cdot), T \times T \times T \sum \times U \rightarrow \sum \left(S(t, t_0, x_0, u(\cdot))|_{t_0}^t\right) = x(t)$
    - $r :=$ read0out map $r(\cdot, \cdot, \cdot): T \times \sum \times U \rightarrow Y, r(t, x(t), u(t) = y$
    
    The state transition map is required to satisfy two axioms
    1) State transistion axiom
    
    If $u(t) = \tilde{u}(t)$ over $[t_0, t]$, then $S(t, t_0, x_0, u(\cdot)) = S(t, t_0, x_0, \tilde{u}(t))$
    $\implies$ what happens before $t_0$ and after $t_1$ does not affect the state on $[t_0, t]$
    
    2) Semi-group property
    $$
    S(t_1, t_0, x_0, u) = S(t_1, t^*, s(t^*, t_0, x_0, u), u)
    $$
    
    **Summary**$\\$
    A dynamical system $D$ has 5 elements and satisfies two axioms.
    
    The composition of the state transition function and the output function called the response function:
    $$
    y(t) = r(t, x(t), u(t))
    $$
    
    If we define a function$\phi: T \times T \times \sum \times U \rightarrow Y$
    then:
    $$
    y(t) = r(t, s(t, _t0, x_0, u), u(\cdot)) =: \phi(t, t_0, x_0, u)
    $$
    
    #### Linearity
    A dynamical system is linear if $\\$
    $(i)$ $U, \sum, Y$ are all vector spaces ove the field $F$ $\\$
    $(ii)$ The response function satisfies the superposition principle (linear in states + inputs)
    $$
    \phi(t, t_0, \alpha_1 x_0^{(1)} + \alpha_2 x_0^{(2)}, u(t)) = \alpha_1 \phi(t, t_0, x_0^{(1)}, u(t)) + \alpha_2 \phi(t, t_0, x_0^{(2)}, u(t))
    $$
    
    Note:
    $$
    \phi(t, t_0, x_0, u) = \phi(t, t_0, 0_x, u) + \phi(t, t_0, x_0, 0_u)
    $$
    RHS: forced respone + natural response
    
    
    """)
    
    st.divider()
    st.header("Lecture 9")
    st.subheader("Thursday, October 23, 2025")
    st.markdown(r"""
    ### Time-Invariant OS    
    Define the shift operator
    $$
    (T_\tau u)(t) = u(t-\tau)
    $$     
    
    #### Time invariance
    #### 1) U is closed under shifts $T_\tau$
    (If $u(\cdot) \in U \implies (T_\tau u)(\cdot) \in U$)
    
    #### 2) $\phi(t_1, t_0, x_0, u) = \phi(t_1 + \tau, t_0 + \tau, x_0, T_\tau u)$
    
    $$
    \forall t_0, t_1 \geq 0\\
    \forall \text{ shifts } \tau\\
    \forall x_0 \in \Sigma\\
    \forall u \in U
    $$
    
    ---
    #### Linear time-varying systems (Callier + Desoer Ch. 2)
    $$
    R = [A(t), B(t), C(t), D(t)]
    $$
    Stands for 
    $$
    \begin{cases}
    x(t_0) = x_0\\
    \dot{x}(t) = A(t)x(t) + B(t)u(t)\\
    y(t) = C(t)\lambda(t) + D(t) u(t)
    \end{cases}
    $$
    where $\underbrace{x(t) \in R^n}_{\Sigma}, \underbrace{y(t) \in R^{n_0}}_Y, \underbrace{u(t) \in R^{n_i}}_U$
    
    The state transition function is the solution to the system
    
    ---
    ### Existence and Uniqueness of Linear Time Varying Solutions
    Let $A(t), B(t), C(t), D(t)$ be matrix valued piecewise continuous functions and let $u(t) \in U$ be a continuous function from $R_+ \to R^{n_i}$.
    
    Then
    $$
    f(x, t) = A(t)x(t) + B(t) u(t) = \dot{x}(t)
    $$
    is a piecewise function in time.
    
    Is $f(\cdot, t)$ Lipschitz continuous in $x$?
    
    $$
    ||f(x, t) - f(y, t)|| = ||A(t)x + B(t)u - A(t)y - B(t)u||\\
    = ||A(t)(x-y)|| = ||A(t)|| \cdot ||x-y||, \quad \forall x, y \in \Sigma
    $$
    
    $k(t)$ is a Lipschitz function since $1) > 0$ and $2)$ is piecewise
    
    $\implies f(\cdot, t)$ is globally Lipschitz in x
    $\implies$ By the fundamental lemma of Differential Equations, $\exists$ a unique solution to Linear time-varying systems
    
    The solution to Linear time-varying systems can be represented by the state transition map
    $$
    x(t) = s(t, t_0, x_0, u(\cdot))
    $$
    and there is a corresponding response map
    $$
    y(t) = \phi(t, t_0, x_0, u(\cdot))
    $$
    
    #### How about linearity?
    $\implies$ check superposition principle from lecture 8
    
    ---
    ### State transition matrix/function
    
    Vector Differential Equation
    $$
    \dot{x}(t) = A(t)x(t) + B(t) u(t)\\
    \dot{X}(t) = A(t)X(t), \quad X(0) = X_0
    $$
    
    Define $\underbrace{\Phi(t_1, t_0)}_{\text{state transition function}}$ as the solutions to MDE with $\Phi(t_0, t_0) = I$
    
    #### Properties
    #### 1) The solution to VDE is given by
    $$
    x(t) = \boxed{\Phi(t, t_0)}x_0
    $$
    
    #### 2) $\Phi(t, t_0) = \Phi(t, t_0)\Phi(t_0, t_0), \quad \forall t, t_0 \in R$
    
    #### 3) $\Phi(t, t_0)^{-1} = \Phi(t_0, t)$
    
    #### Proof of (1):
    $$
    \underbrace{x(t)}_{\text{LHS}} = \underbrace{\Phi(t, t_0)x_0}_{\text{RHS}}
    $$
    
    Strategy prove that both the LHS and RHS satisfy VDE with same IC $\implies$ they must be the same function because of uniqueness of solution to Linear Time Varying systems
    """)
    l9col1, l9col2 = st.columns(2)
    with l9col1:
        st.markdown(r"""
        $$
        \text{LHS}\\
        x(t_0) = x_0\\
        \dot{x}(t) = A(t)x(t) + B(t)u(t)
        $$         
        by definition
        
        $$
        \begin{align*}
        \text{LHS}(t_1) = \Phi(t_1, t_0) \\
        \dot{\text{LHS}}(t) = \dot{\Phi}(t, t_0) \\
        = A(t) \Phi(t, t_0)
        \end{align*}
        $$
        """)
        
    with l9col2:
        st.markdown(r"""
        $$
        \text{RHS}\\
        \begin{align*}
        \Phi(t_0, t_0)x_0 = Ix_0 = x_0 \\
        \frac{d}{dt}\underbrace{[\Phi(t, t_0)x_0]}_{MDE} = \dot{\Phi}(t, t_0)x_0\\
        = A(t) \Phi(t, t_0)x_0\\
        = A(t)\underbrace{(\Phi(t, t_0)x_0)}_{VDE} 
        \end{align*}
        $$
        $\implies$ solution to VDE
        
        $$
        \begin{align*}
        \text{RHS}(t_1) &= \Phi(t_1, t_1)\Phi(t_1, t_0) = \Phi(t_1, t_0) \\
        \dot{\text{RHS}}(t) &= \frac{d}{dt}\big[\Phi(t, t_1)\underbrace{\Phi(t_1, t_0)}_{\text{constant}}\big] \\
        &= \dot{\Phi}(t, t_1) \cdot \Phi(t_1, t_0) \\
        &= A(t)\left(\Phi(t_1, t_1) \cdot \Phi(t_1, t_0)\right)
        \end{align*}
        $$
        $\implies$ also a solution!
        """)
        
    st.markdown(r"""
    $\implies$ LHS and RHS satisfy the same Differential Equation and match at an arbitrary point $\implies \square$         
    """)
    
if section == "Week 5":
    st.title("Week 5")
    st.header("Lecture 10:")
    st.subheader("Tuesday, October 28, 2025")
    
    st.markdown(r"""
       
     ### Recap
     
     $$
     \dot{X}(t) = A(t) X(t), \quad x(t) \in R^m, A(t) \in R^{m \times m}
     $$        
     
     and $\Phi(t, t_0)$ is the solution to the MDE $X(t)$ with initial condition
     
     $$
     X(t_0) = \Phi(t_0, t_0) = I
     $$
     
     ### Vector Differential Equation
     
     $$
     \dot{x}(t) = A(t) x(t), \quad x(t) \in R^m \\
         x(t_0) = x_0 \\
        x(t) = \Phi(t, t_0)x_0 \quad \text{is the solution to the VDE}
     $$
                
    ### Solutions to forced/non-autonomous VDE
    
    #### Linear Time Varying Case
    $$
    \underbrace{\dot{x}(t)}_{n\times 1} = \underbrace{A(t)}_{n\times n} \underbrace{x(t)}_{n \times 1} + \underbrace{B(t)}_{n \times n_i} \underbrace{u(t)}_{n_i \times t}, \quad x(t_0) = x_0
    $$
    
    How do solutions to this problem look like?
    
    **Theorem:**
    
    The solution is given by 
    $$
    \boxed{
    x(t) = \Phi(t, t_0)x_0 + \int_{t_0}^t \Phi(t, \tau) B(\tau)u(\tau)d\tau
    }
    $$
    
    **Proof:**
    
    1) Initial Condition: 
    $$
    \begin{array}{l}
    x(t_0) = \Phi(t_0, t_0) + \int_{t_0}^{t_0} [f(t_0, \tau)=0]B(\tau)u(\tau)d\tau\\
        = I \cdot x_0\\
        = x_0 \quad \square
    \end{array}
    $$
    
    2) Differential Equation: show $x(t)$ satisfies the equation
    - Recall Leibniz rule:
    $$
    \frac{\partial}{\partial z} \int_{a(z)}^{b(z)} f(x, z) dx = \frac{\partial b}{\partial z} \cdot f(b, z) - \frac{\partial a}{\partial z} f(a, z) + \int_{a(z)}^{b(z)}\frac{\partial f}{\partial z} dx
    $$
    
    Using this, we get that 
    $$
    \dot{x}(t) = A(t) \Phi(t, t_0)x_0 + \frac{\partial}{\partial t}(t) f(t, t) - [\frac{\partial}{\partial t}(t_0) = 0] + \int_{t_0}^t \frac{\partial f}{\partial t} \partial \tau\\
        = A(t) \Phi(t, t_0)x_0 + \underbrace{\Phi(t, t)}_I B(t)u(t) + \int_{t_0}^{t} A(t)\Phi(t, \tau) B(\tau)u(\tau)d\tau \\
        = A(t)\underbrace{[\Phi(t, t_0)x_0 + \int_{t_0}^t \Phi(t, \tau)B(\tau)d\tau]}_{x(t)} + B(t)u(t)\\
        \implies \dot{x(t)} = A(t)x(t) + B(t) u(t) \quad \square
    $$
    
    ### Jacobian Linearization:
    $$
    \dot{x} = f(x, u, t), \quad x(t_0) = x_0
    $$
    
    Let the input $u^0(\cdot)$ result in the state $x^0(\cdot)$
    
    Now let $u^0(\cdot)$ be perturbed to $u^0(\cdot) + \delta u(\cdot)$ with resultant state perturbation $x^0(\cdot) + \delta x(\cdot)_n$
    
    Also let the initial condition be perturbed to $x_0 + \delta x_0$
    
    $$
    \dot{x}^0 = f(x^0, u^0, t), \quad x^0(t_0) = x_0
    $$
    
    Now consider what happens to the perturbation
    $$
    \dot{x}^0 + \delta \dot{x} = f(x^0 + \delta x, u^0 + \delta u, t), \quad x^0 + \delta x(t_0) = x_0 + \delta x_0
    $$
    
    By Taylor Expansion around $(x^0, u^0)$:
    
    $$
    f(x^0 + \delta x, u^0) + \underbrace{\frac{\partial}{\partial x}f(x, u, t)|_{x^0, u^0}}_{\text{Jacobian wrt }x}\cdot \delta x + \underbrace{\frac{\partial}{\partial u} f(x, u, t)|_(x^0, u^0)}_{\text{Jacobian wrt }u} \cdot \delta u + H.O.T
    $$
    *higher order terms are assumed to go to zero since $\delta x, \delta u$ are small so $(\delta x)^2, (\delta u)^2$ are very small
    
    $$
    \dot{(\delta x)} = \underbrace{D_x f |_{x^0, u^0}}_{n\times n: A(t)} \cdot \delta x + \underbrace{D_u f|_{x^0, u^0}}_{n\times n_i: B(t)} \cdot \delta u
    $$
    
    #### Example: Pendulum
    $$
    \theta, m, l, \tau\\
    ml^2 \ddot{\theta} - mgl sin \theta = \tau \\
    $$
    Introduce: $x_1 = \theta, x_2 = \dot{\theta}$
    $$
    \implies ml^2 \dot{x}_2 - mgl sin(x_1) = \tau\\
    \dot{x}_2 = \frac{g}{l} sin(x_1) + \underbrace{\frac{\tau}{ml^2}}_u
    $$
    define $\Omega^2 = \frac{g}{l}$
    $$
    = \Omega^2 sin(x_1) + u\\
    \implies \dot{x} = f(x_1, u) = \begin{bmatrix} x_2 \\ \Omega^2 sin(x_1) + u \end{bmatrix}
    $$
    
    **Linearization around vertical position:** 
    $$
    (x_1^0 = 0, \tau^0 = 0)
    $$
    
    $$
    \implies \dot{(\delta x)} = \begin{bmatrix} 0 & 1 \\ \Omega^2 cos(x_1) & 0 \end{bmatrix}|_{0,0}\cdot \delta x + \begin{bmatrix} 0 \\1 \end{bmatrix} \cdot \delta u\\
    \implies \dot{(\delta x)} = \begin{bmatrix} 0 & 1 \\ \Omega^2 & 0\end{bmatrix} \delta x+ \begin{bmatrix} 0 \\ 1 \end{bmatrix} \delta u
    $$
    
    ### Summary
    - Showing that the solution to the MDE is the solution to the VDE (specific form)
    - Given by "theorem"
    - Prove it actually satisfies the linear time varying vector differential equation by Leibniz rule
    - Did linearization looking at a non linear system and now understand linear time varying systems as linearizations of non linear systems
    - Gave example of how linearization looks like
    """)
    
    st.divider()
    st.header("Lecture 11:")
    st.subheader("Thursday, October 30, 2025")
    st.markdown(r"""
    ### Matrix Exponential
    
    #### Linear Time Invariant Case
    $$
    \dot{x}(t) = Ax(t) + Bu(t)
    $$
    
    **Claim:** the state transition matrix for $\dot{x}(t) = A(t) x(t)$ is 
    $$
    \Phi(t, t_0)=exp(A(t-t_0))
    $$
    
    where
    $$
    \underbrace{exp(At)}_{n \times n} = \underbrace{I}_{n \times n} + \underbrace{A}_{n \times n \cdot 1}t + \frac{A^2}{2!}t^2 + \frac{A^3}{3!}t^3 + \dots 
    $$
    
    **Proof:**
    We know that the solution to $\dot{x} = Ax, x(t_0) = x_0$ is $x(t) = \Phi(t, t_0)x_0$
    $$
    \implies x(t) = exp(A(t-t_0))x_0
    $$
    then $x(t_0) = exp(A \cdot (t_0 - t_0))x_0 = exp(0)\cdot x_0 = x_0$
    
    #### Does it satisfy the Differential Equation?
    $$
    \dot{x}(t) = \frac{d}{dt}[exp(A(t-t_0))x_0] = \frac{d}{dt}\left( I + A(t-t_0) + \frac{A^2}{2!}(t-t_0)^2 + \dots\right)x_0\\
    = [0 + A + A(t-t_0)t + \frac{A^3}{2!}(t-t_0)^2 + \dots]x_0\\
    = A[I + A(t-t_0)+\frac{A^2}{2!}(t-t_0)^2 + \dots]x_0\\
    = A[exp(A(t-t_0))]x_0 = Ax(t)
    $$
    
    ### Properties of $e^{At}$:
    
    1) $e^0 = I_{n \times n}$
    2) $e^{A(t+s)} = e^{At} \cdot e^{As}$
    3) $e^{(A+B)t} = e^{At} \cdot e^{Bt} \iff AB = BA$
    4) $(e^{At})^{-1} = e^{A(-t)} = e^{-At}$
    5) $\frac{d}{dt} e^{At} = A \cdot e^{At}$
    6) $X(t) \in R^{n \times n} \rightarrow \dot{X}(t) = AX(t), X(0)= I$ the solution is given by $X(t) = e^{At}$
    
    #### Computing $e^{At}$:
    
    > "26 dubious ways of computing the matrix exponential!"
    
    #### 1) Power Series (only if $A$ is nonpotent, but in general a bad iea numerically; only good for analysis)
    $$
    A = \begin{bmatrix}0 & 1\\0 & 0\end{bmatrix} \rightarrow A^2 = \begin{bmatrix}0 & 0 \\0 & 0\end{bmatrix} \implies e^{At} = I + At + 0 \dots
    $$
    - good idea if $A$ has structure which allows infinite series to become finite
    $$
    = \begin{bmatrix} 1 & t \\0 & 1\end{bmatrix}
    $$
    
    #### 2) $\begin{cases}\dot{x}(t) = Ax(t), \quad x(t) \in R^{n \times n}\\ X(0) = I \end{cases}$
    $$
    L\{f\}(s) = \int_0^\infty f(t)e^{-st}dt\quad t\in R_t, s \in C
    L(\cdot) = s\hat{X}(s) - \hat{x}(0) = A\hat{X}(s) \\
    \iff(sI - A)\hat{X}(s) = x(0) = I\\
    \iff \hat{x}(s) = (sI - A)^{-1}\\
    \rightarrow X(t) = L^{-1}\{(sI -A)^{-1}\}
    $$
    and we know that
    $$
    X(t) = e^{At}\\
    \implies e^{At} = L^{-1}\{(sI - A)^{-1}\}
    $$
    
    **Example**
    $$
    A = \begin{bmatrix}0 & 1\\0 & 0\end{bmatrix}
    $$
    Then: $(sI - A) = \begin{bmatrix}s & -1\\0 & s\end{bmatrix}$
    $$
    (sI - A)^{-1} = \frac{1}{s^2}\begin{bmatrix}s & +1 \\0 & s\end{bmatrix} = \begin{bmatrix}\frac{1}{s} & \frac{1}{s^2} \\ 0 & \frac{1}{s}\end{bmatrix}
    $$
    $$
    \implies L^{-1}\{(sI - A)^{-1}\} = L^{-1}\{\begin{bmatrix}\frac{1}{s} & \frac{1}{s^2} \\ 0 & \frac{1}{s}\end{bmatrix}\} = \begin{bmatrix}1 & t \\0 & 1\end{bmatrix}
    $$
    
    #### Recall: 
    $$
    \dot{x} = Ax + Bu, x(t_0) = x_0, x \in R^n\\
    y = Cx + Du
    $$
    $$
    x(t) = \Phi(t, t_0)x_0 + \int_{t_0}^t \Phi(t, \tau)B u(\tau)d\tau\\
        = e^{A(t-t_0)}x_0 + \int_{t_0}^t e^{A(t - \tau)}Bu(\tau)d\tau
    $$
    
    #### Cayley - Hamilton Theorem
    $$
    (sI - A)^{-1} = \frac{\text{Adjugate}(sI - A)}{\text{det}(sI - A)}
    $$
    
    **Example**: $\begin{bmatrix}a & b \\c & c\end{bmatrix} = \frac{1}{ad - bc} \cdot \begin{bmatrix}d & -b \\-c & a\end{bmatrix}$
    
    $$
    det(sI - A) = s^n + d_1 s^{n-1} + d_2 s^{n-2} + \dots d_n\\
    = X_A(s) = \text{characteristic polynomial of }A
    $$
    
    $$
    \text{Adjugate}(sI - A) = B_0 s^{n-1} + B_1 s^{n-2} + \dots + B_{n-1}
    $$
    $B_i$ are $n \times n$ matrices
    
    **Cayleight- Hamilton Theorem**:$\\$
    $$
    \text{Every matrix A satisfies its own characteristic polynomial}
    $$
    $$
    X_A(A) = A^n + d_1 A^{n-1} + d_2 A^{n-2} + \dots + d_n I = O_{n\times n}
    $$
    
    **Note:** Let $\hat{p}_1(s), \hat{p}_2(s)$ be two polynomials in $s$/
    
    **Then:** 
    $$
    \frac{\hat{p}_1(s)}{X_A(s)} = \hat{q}_1(s) + \frac{\hat{r}_1(s)}{X_A(s)} \rightarrow \hat{p}_1(s) = \hat{q}_1(s)X_A(s) + \hat{r}_1(s)\\
    \frac{\hat{p}_2(s)}{X_A(s)} = \hat{q}_2(s) + \frac{\hat{r}_2(s)}{X_A(s)} \rightarrow \hat{p}_2(s) = \hat{q}_2(s)X_A(s) + \hat{r}_2(s)
    $$
    Even if $\hat{p}_1(s) \neq \hat{p}_2(s)$, if $\hat{r}_1(s) = \hat{r}_2(s)$ then
    $$
    \hat{p}_1(A) = \hat{p}_2(A) \implies \text{polynomials evaluated in A are the same}
    $$
    
    To see this:
    $$
    \hat{p}_1 = \hat{q}_1(A)[X_A(s) = 0] + \hat{r}_1(A) \quad \rightarrow \hat{p}_1(A) = \hat{p}_2(A) \\
    \hat{p}_2 = \hat{q}_2(A)[X_A(s) = 0]  + \hat{r}_2(A) \quad \text{                      if } \hat{r}_1 = \hat{r}_2
    $$
    
    This implies that every polynomial function of $A$ can be written as a function of powers of $A$, up to the order $A^{n-1}$ which means
    a polynomial in the $I, A, A^2, \dots, A^{n-1}$
    
    #### Initial Conditions and Eigenvectors
    $$
    \dot{x} = Ax, \quad x_0, \quad \text{we have }x(t) = e^{At}x_0
    $$
    
    Lets assume $x_0 = \alpha_i(0)v_i$
    $$
    Av_i = \lambda_i v_i
    \implies x(t) = e^{At} \cdot \alpha_i v_i\\
    = (I + At + \frac{A^2t^2}{2!} + \frac{A^3 t^3}{3!} + \dots)\alpha_i v_i\\
    = (v_i + \lambda_i v_i t + \frac{1}{2!} \lambda_i^2 t_2 v_i + \dots) \alpha_i\\
    = (1 + \lambda_i t + \frac{\lambda_i^2}{2!}t^2 \frac{\lambda_i^3}{3!}t^3 + \dots)\alpha_i v_i\\
    = \underbrace{e^{\lambda_i t}}_{\text{scalar}}\cdot \underbrace{\alpha_i v_i}_{x(0) \text{eigenvectors}}
    $$
    #### $e^{At}$ and diagonalization of $A$
    
    $$
    A = VDV^{-1}
    \implies e^{At} = exp(VDV^{-1})\\
    = I + (VDV^{-1})t + (VDV^{-1})^2\frac{t}{2!} + (VDV^{-1})^3 \frac{t^3}{3!} + \dots\\
    = V(I + D + D^2 \frac{t^2}{2!} + D^3\frac{t^3}{3!})V^{-1}\\
    = Ve^{Dt}V^{-1}
    $$
    much easier to compute, can even use power series here
    
    **Also Note:** 
    $$
    e^{Dt} = L^{-1}\{(sI-D)^{-1}\}\\
    = L^{-1}\{\begin{bmatrix} s-\lambda_1 & & \\ & \ddots & \\ & & s-\lambda_m\end{bmatrix}^{-1}\}\\
    = L^{-1}\{\begin{bmatrix} \frac{1}{s-\lambda_1} & & \\ & \ddots & \\ & & \frac{1}{s-\lambda_m}\end{bmatrix}\}\\
    \implies e^{At} = V \text{diag}(e^{\lambda_i t}) V^{-1}
    $$    
    
    """)
    