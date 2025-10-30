import streamlit as st
import matplotlib.pyplot as plt

st.title("MAE 280A Linear Systems Theory")

section = st.selectbox(
    "",
    [
        "Week 1",
        "Week 2",
        "Week 3",
        "Week 4"
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
        """)
    
    # fig, ax = plt.subplots(figsize=(6, 4))
    
    # st.pyplot(fig)
    
if section == "Week 2":
    st.title("Week 2")
    
    st.header("Lecture 4: Absent")
    st.subheader("Tuesday, October 07, 2025")

    st.divider()
    st.header("Lecture 5: Absent")
    st.subheader("Thursday, October 09, 2025")
    
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