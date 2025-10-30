import streamlit as st
import matplotlib.pyplot as plt

st.title("ECE 271A Statistical Learning")

section = st.selectbox(
    "",
    [
        "Introduction",
        "Bayesian Decision Theory",
        "Gaussian Classifier",
        "Maximum Likelihood Estimation"
    ],
)

if section == "Introduction":
    st.title("Introduction")
    st.header("Lecture 1 - Monday, 9/29/25")
    st.markdown(r"""
       
    #### Supervised Learning
    - $Y \in {0,1}$: detection
    - $Y \in {0, \dots, n}$: classification
    - $Y$ is real and continuous: regression
    
    ##### **Classification**
    - discriminative: learns decision boundary
    - generative: learns how data is generated
    
    In this class we will focus more on generative methods
    
    ##### Overfitting
    - only test results matter
    - we care about generalization: accuracy outside training set
           
    """)

if section == "Bayesian Decision Theory":
    st.title("Bayesian Decision Theory")
    st.header("Lecture 2: Wednesday, 10/01/25")
    
    # Radio selector for main sections
    subsection = st.radio(
        "Bayesian decision problems solve optimal decisions involving uncertainty    ",
        ["Bayesian Decision Theory", "Probabilistic Representations and Inference"]
    )

    if subsection == "Bayesian Decision Theory":
        st.markdown("### Bayesian Decision Theory")
        st.markdown(r"""
        - **World**: has states drawn from a state or class random variable $Y$
        - **Observer**: measures observations (features), drawn from a random process $X$
        - **Decision Function**: observer uses the observations to make decisions about the state of the world $y$
        $$
        x = \Omega \quad y = \Psi
        $$
        
        $$
        g: \Omega \rightarrow \Psi \quad g(x) = y
        $$
        - **Loss function**: the cost of deciding for $\hat{y}$ when the true state is $y$
        - **Goal** determine the optimal decision function for the loss
        
        #### Classification
        $$
        g(x) = i, \quad i \in [1, \dots N]
        $$
        - **$0-1$ loss function**
        
        $$
        L[g(x), y] = \begin{cases} 1, \quad g(x) \neq y \\ 0, \quad g(x) = y\end{cases}
        $$
        
        #### Regression
        $$
        g(x) \in \mathscr{R}
        $$
        - uses a suitable loss function like the square error
        $$
        L[g(x), y] = ||y - g(x)||^2
        $$
        
        #### Goal
        - We learn decision functions that are optimal on average
        """)

    elif subsection == "Probabilistic Representations and Inference":
        st.markdown(r"""
        #### Probabilistic representations
        - **joint distribution**
            $$
            P_{X, Y}(x, i)
            $$
            
            $$
            P_{X, Y}(x, i) = P_{X, Y}(x|i)P_Y(i)
            $$
        
        - $P_{X, Y}$ is the **class conditional distribution**
        - $P_Y(i)$ is the class probability
        
        #### Properties of probabilistic inference
        - **Chain rule of probability**
            $$
            P(x, y) = P(x|y)P(y)
            $$
            
            $$
            P(x_1, x_2, \dots, x_n) = P(x_1| x_2, \dots x_n) \times P(x_2, | x_3, \dots x_n) \times P(x_{n-1} | x_n)P(x_n)
            $$
            - it allows us to modularize inference problems
            - "The probability of one variable given all the others"
        - **Marginalization**
            - Frequently we have problems with multiple random variables - But frequently we only care about a subset of $X$ observation vector
            - We marginalize with respect to a sbset of variables
            - This is done by summing (or integrating) the others out
            $$
            P(x_1, x_4) = \sum_{x_2, x_3} P(x_1, x_2, x_3, x_4)
            $$
            - Important with big models as most variables are irrelevant
            - Major field is **dimensionality reduction** 
            - We combine marginalization with the chain rule to explore independence relationships that will allow us to reduce computation
        - **Independence**
            - $X$ and $Y$ are independent variables if the joint distribution has no more information than the marginal distribution
            $$
            P(x, y) = P(x | y)P(y) = P(x)
            $$
            - Example: Being sick is independent of someone shivering as soon as you measure their temperature as a better feature
        - **Bayes Rule**
    """)

    st.divider()
    st.header("Lecture 3: Monday, 10/06/25")
    
    l3_subsection = st.radio(
        "Topics",
        ["Bayes Rule", "Bayes Decision Rule", "Example"]
    )
    if l3_subsection == "Bayes Rule":
        st.subheader("Bayes Rule")
        st.markdown(r"""
        $$
        P(y|x) = \frac{P(x|y)P(y)}{P(x)}
        $$
        
        - allows us to "switch" the relation between the variables
        - this is very complicated because it is not causal
        - we are asking for the probability of cause given consequence
                 
        - e.g for medical diagnosis
        $$
        P(\text{disease}| \text{symptom})
        $$
        
        - Bayes rule transforms it into the probabililty of consequence given the cause
        
        $$
        P(\text{disease| \text{symptom}}) = \frac{P(\text{symptom}| \text{disease})P(\text{disease})}{P(\text{symptom})}
        $$
        - note that $P(\text{symptom}|\text{disease})$ is very easy as you can get it out of any medical textbook
        - $P(\text{disease y})$ does note depend on the patient, you can get it by collecting statistics over the entire population and is known as the **prior** probability
        - $P(\text{symptom x})$ is a combination of the two (marginalization) and is the constant that normalizes the probability
        $$
        P(\text{symptom x}) = \sum_y P(\text{symptom x | disease y}) P(\text{disease y})
        $$
        where $y$ in this case is "all diseases
    
        """)
        
    if l3_subsection == "Bayes Decision Rule":
        st.subheader("Risk")
        st.markdown(r"""
        #### recall that we have
        - $Y$: state of the world
        - $X$: observations
        - $g(x)$: decision function
        - $L[g(x), y]$: loss of predicting $y$ with $g(x)$
        
        #### the expected value of the loss is called the risk
        $$
        \text{Risk} = E[L(X, Y)]
        $$
        
        $$
        \text{Risk} = \int \sum_{i = 1}^M P(i, x) L[g(x), i], dx = E[R(x)]
        $$
        
        - the sample average approaches this expectation given the **Law of Large Numbers**
        $$
        \hat{\text{Risk}} = \frac{1}{N} \sum^N L[g(x), y]
        $$
        
        - from this
        $$
        \text{Risk} = \int \sum_{i = 1}^M P(i, x) L[g(x), i], dx 
        $$
        - by chain rule
        $$
        \text{Risk} = \int P(x)\sum_{i = 1}^M P(i | x) L[g(x), i], dx 
        = \int P(x) R(x), dx = E[R(x)]
        $$
        
        - where
        $$
        R(x) = \sum_{x=1}^M P(i|x)L[g(x), i]
        $$
        
        - is the conditional risk given the observation $x$
        
        - since by definition
        $$
        L[g(x), i] \geq 0
        $$
        
        - it follows that
        $$
        R(x) = \sum_{i=1}^M P(i|x)L[g(x), i] \geq 0
        $$
        
        - Also
        $$
        \text{Risk} = E[R(X)]
        $$
        - is minimum if we minimize $R(X)$ at all $x$ i.e. if we use pick the decision function
        $$
        g^*(x) = \text{arg min}_{g(x)}\sum_{i=1}^M P(i|x)L[g(x), i]
        $$
        
        ### Bayes Decision Rule
        
        - this is called the **Bayes Decision Rule**
        
        $$
        g^*(x) = \text{arg min}_{g(x)}\sum_{i=1}^M P(i|x)L[g(x), i]
        $$
        
        - The asssociated risk
        $$
        R^* = \int \sum_{i = 1}^M P(i|x)L[g^*(x), i)], dx
        $$
        
        - or
        $$
        R^* = \int P(x) \sum_{i = 1}^M P(i|x)L[g^*(x), i)], dx
        $$
        
        - This is the Bayes risk, and it cannot be beaten
        
        
        """)
    
    if l3_subsection == "Example":
        st.subheader("Example")
        st.markdown(r"""
        Lets consider a binary classification problem
        

        $$
        g^*(x) \in [0, 1]
        $$       
        
        For which the conditional risk is:
        $$
        R(x) = \sum_{i = 0}^1 P(i|x)L[g(x), i] \\
        = 
        $$  
        
        Pick the class with the least risk
        
        i.e. pick $g(x) = 0$ if $R_0(x) < R_1(x)$
        
        This is the same as
        
        $$
        P(0|x)L[1, 0] > P(1|x)L[0,1]
        $$
        
        **Pick 0**
        $$
        \frac{P(0|x)}{P(1|x)} > \frac{L[0,1]}{L[1,0]}
        $$
        
        - Applying Bayes Rule
        $$
        \frac{P(x|0)P_y(0)}{P(x|1)P_y(1)} > \frac{L[0,1]}{L[1,0]}
        $$
        
         $$
        \frac{P(x|0)}{P(x|1)} = T^* > \frac{L[0,1]P_y(1)}{L[1,0])P_y(0)}
        $$
        
        - i.e.: we pick 0 when the probability of $X$ given that $Y=0$ divided by that given $Y=1$ is greatre than a threshold
        
        The optimal threshold depends on the costs of the two types of eror and the probabilities of the two claases
        
        ... see slides
        """)

if section == "Gaussian Classifier":
    st.title("Gaussian Classifier")
    st.header("Lecture 4 - Wendesday, 10/08/25")
    
    l4_sub = st.radio(
        "Topics",
        ["Bayesian Decision Theory", "MAP Rule"]
    )
    
    if l4_sub == "Bayesian Decision Theory":
        st.markdown(r"""
        ### Bayesian Decision Theory
        
        **Recall that we have:**
        - $Y$: state of the world
        - $X$: observations
        - $g(x)$: decision function
        - $L[g(x), y]$: loss of predicting y with $g(x)$
        
        **Bayes decision rule** is that the rule minimizes the risk 
        $$
        \text{Risk} = E[L(X, Y)]
        $$
        
        Given x, it consists of picking the prediction of **minimal conditional risk**
        $$
        g^*(x) = \text{arg min}_{g(x)} \sum_{i = 1}^M P(i|x)L[g(x), i]
        $$
        
        """)
        
    if l4_sub == "MAP Rule":
        st.markdown(r"""
        
        For the $0-1$ loss
        $$
        L[g(x), y] = \begin{cases} 1, \quad g(x) \neq y \\ 0, \quad g(x) = y \end{cases}
        $$
        
        The optimal decision rule is the **maximum a-posteriori** probability rule
        $$
        g^*(x) = \text{arg max}_i P(i|x)
        $$
        
        The **associated risk** is the probability of error of this rule (**Bayes error**)
        
        There is no other decision function with lower error
        
        By application of simple mathematical laws (**Bayes Rule, monotonicity of the log**)
        
        We have shown that the following **Three decision rules** are optimal and equivalent
        
        1. $i(x) = \text{arg max}_i P(i|x)$
        2. $i^*(x)  = \text{arg max}_i [P(x|i)P(i)]$
        3. $i^*(x) = \text{arg max}_i [\text{log} P(x|i) + log P(i)]$
        
        $1)$ is usually hard to use, $3)$ is freqently easier that $2)$
                 
        """)
    
    st.header("Lecture 5 - Monday, 10/13/25")
    l5_sub = st.radio(
        "Topics",
        ["MAP Rule", "BDR", "The Gaussian Classifier"]
    )
    if l5_sub == "MAP Rule":
        st.markdown(r"""
        
        #### In Summary
        Under the zero-one loss, then the Bayes decision rule reduces to the MAP (Maximum Aposterior Probability) rule.
        
        For the zero/one loss, the following three decision rules are optimal and equivalent
        1) $i^*(x) = \text{argmax}_i P_{Y|X}(i|x)$
        2) $i^*(x) = \text{argmax}_i[P_{X|Y}(x|i)P_Y(i)]$
        3) $i^*(x) = \text{argmax}_i[\text{log} P_{X|Y}(x|i)+\text{log}P_Y(i)]\\$
        
        $1)$ is hard to use, $3)$ is frequently easier than $2)$
        
        #### Example
        - a bit is transmitted by a source, corrupted by noise and recieved by a decoder
        - What should the optimal decoder do to recover $Y$?
            - Just threshold the value
            $$
            Y = \begin{cases}0, \quad \text{if }x < T\\1, \quad \text{if }x > T\end{cases}
            $$
            - What is the threshold? $\rightarrow$ solve with Bayes Decision Rule
        
        **We need**
        - Class Probabilities
        $$
        P_Y(0) = P_Y(1) = \frac{1}{2}
        $$
        - Class-conditional densities
            - Noise results from thermal processes
            - A lot of independent events add up
            - By the central limit theorem, it is reasonable to assume **Gaussian** Noise
        - Gausian Probability density function
        $$
        P_X(x) = G(x, \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
        $$
        - Since the noise is gaussian, and asssuming it is just added to the signal, we have:
        $$
        X = Y + \epsilon \quad \epsilon ~ N(0, \sigma^2)
        $$
        - In both cases, $X$ corresponds to a constant $(Y)$ plus zero-mean Gaussian noise
        - This simply adds $Y$ to the mean of the Gaussian making the mean either $0$ or $1$
        
        In summary
        $$
        P_{X|Y}(x|0) = G(x, 0, \sigma)\\
        P_{X|Y}(x|1) = G(x, 1, \sigma)\\
        P_Y(0) = P_Y(1) = \frac{1}{2}
        $$
        
        #### Example
        Now we can implement an algorithm. To compute the BDR, we recall that
        $$
        i^*(x) = \text{arg max}_i[\text{log}P_{X|Y}(x|i) + \text{log}P_Y(i)]
        $$
        
        and note that
        - terms which are constant can be dropped
        - since we are looking for the $i$ that maximizes the function
        - since this is the case for the class probabilities
        $$
        P_Y(0) = P_Y(1) = 1/2
        $$
        - we have $i^*(x) = \text{argmax} \text{ log} P_{X|Y}(x|i)$
        """)
    
    if l5_sub == "BDR":
        st.markdown(r"""
        ### BDR
        
        Lets consider the more general case
        
        $$
        P_{X|Y}(x|0) = G(x, \mu_0, \sigma) \quad P_{X|Y}(x|1) = G(x, \mu_1, \sigma)
        $$
        
        For which
        $$
        i^*(x) = \text{arg max}_i \text{ log} P_{X|Y}(x|i) \\
        = \text{arg max}_i \text{ log}\left(\frac{1}{\sqrt{2 \pi \sigma^2}} e^{\frac{(x-\mu_i)^2}{2 \sigma^2}}\right)\\
        = \text{arg max}_i \left(-\frac{1}{2} \text{log}(2\pi\sigma^2) - \frac{(x-\mu_i)^2}{2\sigma^2}\right)\\
        = \text{arg min}_i \frac{(x-\mu_i)^2}{2\sigma^2}
        $$
        
        or
        $$
        i^* = \text{arg min}_i \frac{(x-\mu_i)^2}{2\sigma^2}\\
            = \text{arg min}_i(x^2 - 2x\mu_i + \mu_i^2)\\
            = \text{arg min}_i(-2x\mu_i + \mu_i^2)
        $$
        
        
        #### Optimal Decision Rule
        
        - **Pick 0** if
        $$
        -2x\mu_0 + \mu_0^2 < -2x\mu_1 + \mu_1^2
        $$
        $$
        2x(\mu_1 - \mu_0) < \mu_i^2 - \mu_0^2
        $$
        - Or **Pick 0** if
        $$
        x < \frac{\mu_1 + \mu_0}{2}
        $$
        
        #### What is the Point?
        - Can't do higher dimensions visually so we need math
        - The Bayesian solution keeps us honest
        - It forces us to make all our assumptions explicit
        
        #### Assumptions we have made
        - Uniform class probabilities $\quad P_Y(0) = P_Y(1) = 1/2$
        - Gaussianity $\quad P_{X|Y}(x|i) = G(x, \mu, \sigma)$
        - The variance is the same under the two states $\quad \sigma_i = \sigma$
        - Noise is additive $X = Y + \epsilon$
        
        Even for a trivial problem, we have made lots of assumptions
        
        #### What is the role of the prior for class probabilities?
        $$
        x < \frac{\mu_1 + \mu_0}{2} + \frac{\sigma^2}{\mu_1 - \mu_0}\text{log }\frac{P_Y(0)}{P_Y(1)}
        $$
        - The prior moves the threshold up and down 
        - $P_Y(0) > P_Y(0)$: threshold increases
        - Since 0 has a higher probability, we care more about errors on the 0 side
        - By using a higher threshold, we are making it more likely to pick 0
        - If $P_Y(0)=1$, all we care about is $Y=0$, the threshold becomes infinite
        - We never say 1
        
        **How relevant is the prior?**
        - It is weighted by:
        $$
        1/(\frac{\mu_1 - \mu_0}{\sigma^2})
        $$
        - If the distance between the gaussians increases, the prior knowledge becomes less and less important
        - If the gaussians are the same aka $\mu_1 = \mu_0$, then we can ignore the observations entirely as they are impossible to distinguish and contribute no information.
        """)
        
    if l5_sub == "The Gaussian Classifier":
        st.markdown(r"""
        
        This is one example of a Gaussian Classifier
        - In practice, we rarely only have one variable
        - Typically $X = (X_1, \dots, X_d)$ is a vector of observations
        
        The BDR for this case is equivalent, but more interesting$\\$
        
        The main difference is in the class-conditional distributions, which are multivariate Gaussian
        $$
        P_{X|Y}(x|i)= \frac{1}{\sqrt{(2\pi)^d |\sum_i|}}\text{exp}\left({-\frac{1}{2}(x-\mu_i)^T \sum_i^{-1}(x-\mu_i)}\right)
        $$
        
        where $\sum$ is the covariance matrix
        
        In this case
        $$
        P_{X|Y}(x|i) = \frac{1}{\sqrt{(2\pi)^d |\sum_i|}}\text{exp}\left({-\frac{1}{2}(x-\mu_i)^T \sum_i^{-1}(x-\mu_i)}\right)
        $$
        - The BDR
        $$
        i^*(x) = \text{arg max}_i [\text{log}P_{X|Y}(x|i) + \text{log}P_Y(i)]
        $$
        - Becomes:
        $$
        i^*(x) = \text{arg max}_i [-\frac{1}{2}(X - \mu_i)^T \Sigma_i^{-1}(X - \mu_i) - \frac{1}{2}\text{log}(2\pi)^d |\sum_i| + \text{log}P_Y(i)]
        $$
        
        This can be written as
        $$
        i^*(x) = \text{arg min}_i [d_i(X, \mu_i) + \alpha_i]
        $$
        with
        $$
        d_i(x, y) = (x-y)^T\Sigma_i^{-1}(x-y)\\
        \alpha_i = \text{log}(2\pi)^d|\Sigma_i| - 2\text{log}P_Y(i)
        $$
        - The optimal rule is to asssign $x$ to the closest classs
        - Closest is measured with the Mahalanobis distance $d_i(x, y)$
        - To which $\alpha$ constant is added to account for the class prior
        
        #### Special case of interest
        
        Classes have the same covariance
        $$
        \Sigma_i = \Sigma
        $$
        for all $i$
        
        The BDR becomes
        $$
        i^*(x) = \text{arg min}_i [d(x, \mu_i) + \alpha_i]
        $$
        
        with
        
        $$
        d(x, y) = (x-y)^T \sum^{-1}(x-y)\\
        \alpha_i = \text{log}|\Sigma| - 2\text{log}P_Y(i)
        $$
        
        In detail
        $$
        i^*(x) = \text{arg min}_i\left[(x-\mu_i)^T\Sigma^{-1}(x-\mu_i) - 2\text{log}P_Y(i)\right]\\
        = \text{argmax}\left[\mu_i^T \Sigma^{-1}x - \frac{1}{2}\mu_i^T\Sigma^{-1}\mu_i + \text{log}P_Y(i)\right]
        = \text{argmax}_i [w_i^T - w_{i0}]
        $$
        - The covariance affects the geometry of the distributions in the space (stretch, distance, etc)
        
        
        """)
    
    st.header("Lecture 6 - Wednesday 10/15/25 ")
    l6_sub = st.radio(
        "Topics",
        ["Geometric Interpretation of Gaussian Classifier", ]
    )
    if l6_sub == "Geometric Interpretation of Gaussian Classifier":
        st.markdown(r"""
        In summary
        
        $$
        i^*(x) = \text{arg max}_i g_i(x)\\
        g_i(x) = w_i^T x + w_{i0}\\
        w_i = \Sigma^{-1}\mu_i\\
        w_{i0} = -\frac{1}{2}\mu_i^T \Sigma^{-1}\mu_i + \text{log}P_Y(i)
        $$    
        - The BDR is a linear function or a linear discriminant  
        
        #### Geometric Interpretations
        
        Classes $i, j$ share a boundary if 
        - There is a set of $x$ such that
        $$
        g_i(x) = g_j(x)
        $$
        
        - or 
        
        $$
        (w_i - w_j)^T x + (w_{i0} - w_{j0}) = 0\\
        (\Sigma^{-1}\mu_i - \Sigma^{-1}\mu_j^T)x + \left(-\frac{1}{2} \mu_i^T \Sigma^{-1}\mu_i + \text{log}P_Y(i) + \frac{1}{2}\mu_j^T\Sigma^{-1}\mu_j-\text{log}P_Y(j)\right) = 0
        $$
        
        - Can be written as
        $$
        (\mu_i - \mu_j)^T\Sigma x - \frac{1}{2}\left(\mu_i^T\Sigma^{-1} \mu_i - \mu_j^T \Sigma^{-1} \mu_j - 2 \text{log}\frac{P_Y(i)}{P_Y(j)}\right) = 0
        $$
        
        """)
if section == "Maximum Likelihood Estimation":
    st.title("Maximum Likelihood Estimation")
    
    
# regime =  st.radio("", ["Classification", "Regression"])

# if regime == "Classification":
#     st.markdown(r"""
#      ##### **Classification**
# - discriminative
# - generative
#     """)


# st.markdown(r"""
    
# $$
# A = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1\end{bmatrix}
# $$         
            
# """)

# fig, ax = plt.subplots(figsize=(6, 4))

# st.pyplot(fig)