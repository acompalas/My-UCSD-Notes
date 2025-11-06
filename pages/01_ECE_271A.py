import streamlit as st
import matplotlib.pyplot as plt

st.title("ECE 271A Statistical Learning")

section = st.selectbox(
    "",
    [
        "Introduction",
        "Bayesian Decision Theory",
        "Gaussian Classifier",
        "Maximum Likelihood Estimation", 
        "Bias and Variance",
        "Bayesian Parameter Estimation"
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
    
    st.divider()
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
    
    st.divider()
    st.header("Lecture 6 - Wednesday 10/15/25 ")
    l6_sub = st.radio(
        "Topics",
        ["Gaussian Classifier", "Geometric Interpretation of Gaussian Classifier"]
    )
    
    if l6_sub == "Gaussian Classifier":
        st.markdown(r"""
        This is one example of a Gaussian Classifier
        - In practice, we rarely only have one variable
        - Typically $X = (X_1, \dots, X_d)$ is a vector of observations
        
        The BDR for this case is equivalent, but more interesting$\\$
        
        The main difference is in the class-conditional distributions, which are multivariate Gaussian
        $$
        P_{X|Y}(x|i)= \frac{1}{\sqrt{(2\pi)^d |\Sigma_i|}}\text{exp}\left({-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)}\right)
        $$
        
        In this case
        $$
        \boxed{
        P_{X|Y}(x|i) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_i|}}\text{exp}\left({-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)}\right)
        }
        $$
        - The BDR
        $$
        \boxed{
        i^*(x) = \text{arg max}_i [\text{log}P_{X|Y}(x|i) + \text{log}P_Y(i)]
        }
        $$
        - Becomes:
        $$
        \boxed{
        i^*(x) = \text{arg max}_i [-\frac{1}{2}(X - \mu_i)^T \Sigma_i^{-1}(X - \mu_i) - \frac{1}{2}\text{log}(2\pi)^d |\Sigma_i| + \text{log}P_Y(i)]
        }
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
        
        First special case of interest
        - Classes have the same covariance
        - The distance no longer depends on the class
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
        
        #### In summary
        
        $$
        i^*(x) = \text{arg max}_i g_i(x)\\
        g_i(x) = w_i^T x + w_{i0}\\
        w_i = \Sigma^{-1}\mu_i\\
        w_{i0} = -\frac{1}{2}\mu_i^T \Sigma^{-1}\mu_i + \text{log}P_Y(i)
        $$    
        - The BDR is a linear function or a linear discriminant  
        
        """)
    
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
        (\mu_i - \mu_j)^T\Sigma^{-1} x - \frac{1}{2}\left(\mu_i^T\Sigma^{-1} \mu_i - \mu_j^T \Sigma^{-1} \mu_j - 2 \text{log}\frac{P_Y(i)}{P_Y(j)}\right) = 0
        $$
        
        - Next we use
        
        $$
        \mu_i^T \Sigma^{-1} \mu_i - \mu_j\Sigma^{-1}\mu_j = \mu_i^T \Sigma^{-1} \mu_i - \mu_i^T \Sigma^{-1} \mu_j + \mu_i^T \Sigma^{-1} \mu_j - \mu_j^T \Sigma^{-1} \mu_j
        $$
        
        - Which can be written as
        $$
        \mu_i^T(\mu_i - \mu_j) + (\mu_i - \mu_j)^T \Sigma^{-1} \mu_j = \\
        \mu_i^T \Sigma^{-1}(\mu_i - \mu_j) + \mu_j^T \Sigma^{-1} (\mu_i - \mu_j) = \\
        (\mu_i + \mu_j)^T \Sigma^{-1}(\mu_i - \mu_j)
        $$
        
        - using this in 
        $$
        (\mu_i - \mu_j)^T \Sigma^{-1}x - \frac{1}{2} \left(\mu_i^T \Sigma^{-1} \mu_i - \mu_j^T \Sigma^{-1} \mu_j -2 \text{log} \frac{P_Y(i)}{P_Y(j)} \right) = 0
        $$
        
        - Lead to
        $$
        (\mu_i - \mu_j)^T \Sigma^{-1}x - \frac{1}{2} \left((\mu_i + \mu_j)^T \Sigma^{-1} (\mu_i - \mu_j) -2 \text{log} \frac{P_Y(i)}{P_Y(j)} \right) = 0
        $$
        
        $$
        \boxed{
        \begin{array}{l}
        w^T x + b = 0 \\
        w = \Sigma^{-1}(\mu_i - \mu_j) \\
        b = -\frac{(\mu_i + \mu_j)^T \Sigma^{-1}(\mu_i - \mu_j)}{2}
            + \log \frac{P_Y(i)}{P_Y(j)}
        \end{array}
        }
        $$
        - This is the equation of the hyperplane of parameters $w$ and $b$
        
        #### Derive Threshold Function
        $$
        (\mu_i - \mu_j)^T \Sigma^{-1}x - \frac{1}{2} \left((\mu_i + \mu_j)^T \Sigma^{-1} (\mu_i - \mu_j) -2 \text{log} \frac{P_Y(i)}{P_Y(j)} \right) = 0\\
        (\mu_i - \mu_j)^T \Sigma^{-1}\left(x - \frac{(\mu_i + \mu_j)}{2} + \frac{(\mu_i - \mu_j)}{(\mu_i - \mu_j)^T \Sigma^{-1}(\mu_i - \mu_j)}\text{log}\frac{P_Y(i)}{P_Y(j)} \right) = 0
        $$
        
        $$
        \boxed{
        \begin{array}{l}
        w^T(x-x_0) = 0\\
        w = \Sigma^{-1}(\mu_i - \mu_j)\\
        x_0 = \frac{(\mu_i + \mu_j)}{2} - \frac{(\mu_i - \mu_j)}{(\mu_i - \mu_j)^T \Sigma^{-1}(\mu_i - \mu_j)} \text{log}\frac{P_Y(i)}{P_Y(j)}
        \end{array}
        }
        $$
        
        - This is the equation of a hyperplane
            - of normal vector $w$
            - that passes through $x_0$

        It is the optimal decision boundary for Gaussian classes with equal covariance
        
        - Note the similarities with
        $$
        x < \frac{\mu_1 + \mu_0}{2} + \frac{1}{\frac{\mu_1 - \mu_0}{\sigma^2}}\text{log}\frac{P_Y(0)}{P_Y(1)}
        $$
        
        #### First Special Case
        - The covariance is the identity
        $$
        \Sigma = \sigma^2 I
        $$
        
        - Optimal boundary
        $$
        \boxed{
        \begin{array}{l}
        w^T(x-x_0) = 0\\
        w = \frac{\mu_i - \mu_j}{\sigma^2}\\
        x_0 = \frac{\mu_i + \mu_j}{2} - \sigma^2 \frac{(\mu_i - \mu_j)}{||\mu_i - \mu_j||^2}\text{log}\frac{P_Y(i)}{P_Y(j)}\\
        = \frac{\mu_i + \mu_j}{2} - \frac{\sigma^2}{||\mu_i - \mu_j||^2}\text{log}\frac{P_Y(i)}{P_Y(j)}(\mu_i - \mu_j)
        \end{array}
        }
        $$
        
        - What is the effect of the prior? ($P_Y(i) \neq P_Y(j)$)
        
        $x_0$ moves away from $\mu_i$ if $P_Y(i) > P_Y(j)$ making it more likely to pick $i$ and vice versa
        
        #### Second Special Case
        - The covariance is arbitrary
        $$
        \Sigma_i = \Sigma
        $$
        
        - Optimal Boundary
        $$
        \boxed{
        \begin{array}{l}
        w^T(x- x_0) = 0\\
        w = \Sigma^{-1}(\mu_i - \mu_j)\\
        x_0 = \frac{\mu_i + \mu_j}{2} - \frac{1}{(\mu_i - \mu_j)^T \Sigma^{-1} (\mu_i - \mu_j)}\text{log}\frac{P_Y(i)}{P_Y(j)}(\mu_i - \mu_j)
        \end{array}
        }
        $$
        - $x_0$ basically the sam, strength of the prior inversely proportional to the Mahalanobis distance between means
        
        We can show mathematically that the plane is tangent to the pdf iso-contours at $x_0$
        - This reflects the fact that the natural distance is now Mahalanobis
        
        #### Generic Case
        - Covariances are different
        $$
        \boxed{
        \begin{array}{l}
        i^*(x) = \text{arg min}_i [d_i(X, \mu_i) + \alpha_i]\\
        d_i(X, y) = (X - y)^T \Sigma_i^{-1}(X - y)\\
        \alpha_i = \text{log}(2\pi)^d |\Sigma_i|-2\text{log}P_Y(i)
        \end{array}
        }
        $$
        - There is not much to simplify
        $$
        g_i(x) = (x - \mu_i)^T \Sigma_i^{-1}(x - \mu_i) + \text{log}|\Sigma_i| - 2 \text{log}P_Y(i) \\
        = x^T \Sigma_i^{-1}x - 2x^T\Sigma_i^{-1}\mu_i + \mu_i^T\Sigma_i^{-1}\mu_i + \text{log}|\Sigma_i| - 2\text{log}P_Y(i)
        $$
        - Now we can't throw out $x^T \Sigma_i^{-1} x$ because the covariance is not a shared constant
        
        - And
        $$
        g_i(x) = x^T \Sigma_i^{-1}x - 2x^T\Sigma_i^{-1}\mu_i + \mu_i^T \Sigma_i^{-1}\mu_i + \text{log}|\Sigma_i|-2\text{log}P_Y(i)
        $$
        
        - Which can be written as:
        $$
        \boxed{
        \begin{array}{l}
        g_i(x) = x^T W_i x + w_i^T x + w_{i0}\\
        W_i = \Sigma_i^{-1}\\
        w_i = -2 \Sigma_i^{-1}\mu_i\\
        w_{i0} = \mu_i^T \Sigma_i^{-1}\mu_i + \text{log}|\Sigma_i| - 2\text{log}P_Y(i)
        \end{array}
        }
        $$
        - For 2 classes the decision boundary is hyperquadratic (no longer linear)
        - This could mean hyper-plane, pair of hyper-planes, hyper-spheres, hyper-ellipsoids, hyper-hyperboloids, etc
        
        When Linear boundaries no longer work
        1) Use Bayes-Decision Rule (transform the model)
        2) Use a plane but transform the data like a neural network (transform the data)
        
        - We have derived all of this from the log-based BDR
        $$
        i^*(x) = \text{argmax}_i [\text{log} P_{X|Y}(x|i) + \text{log}P_Y(i)]
        $$
        
        - When there are only two classes ,it is also interesting to look at the original definition
        $$
        i^*(x) = \text{argmax}_i g_i(x)
        $$
        
        with
        $$
        g_i(x) = P_{Y|X}(i|x) = \frac{P_{X|Y}(x|i)P_Y(i)}{P_X(x)} =  \frac{P_{X|Y}(x|i)P_Y(i)}{P_{X|Y}(x|0)P_Y(0) + P_{X|Y}(x|1)P_Y(1)}
        $$
        
        If we look at the other form of the BDR, then the connections to neural networks becomes more evident.
        
        #### The sigmoid
        
        - Note that this can be written as
        $$
        i^*(x) = \text{arg max} g_i(x)\\
        g_0 = \frac{1}{1 + \frac{P_{X|Y}(x|1)P_Y(1)}{P_{X|Y}(x|0)P_Y(0)}}\\
        g_1 = 1 - g_0(x)
        $$
        
        - For Gaussian classes, the posterior probabilities are
        $$
        g_0(x) = \frac{1}{1 + \text{exp}\{d_0(x-\mu_0) - d_1(x-\mu_1)+\alpha_0 - \alpha_1\}}
        $$
        - Whereas before
        $$
        d_i(x, y) = (x-y)^T \Sigma_i^{-1}(x-y)\\
        \alpha_i = \text{log}(2\pi)^d |\Sigma_i| - 2\text{log}P_Y(i)
        $$
        Where the posterior is a sigmoid 
        """)
if section == "Maximum Likelihood Estimation":
    st.title("Maximum Likelihood Estimation")
    
    st.header("Lecture 7 - Monday 10/20/25")
    
    l7_sub = st.radio(
        "Topics",
        ["Review", "Maximum Likelihood Estimation" ]
    )
    
    if l7_sub == "Review":
        
        st.markdown(r"""
        ### MAP Rule
        
        We have shown that it can be implemented in any of the three following ways
        - 1) $i^*(x) = \text{argmax}_i P_{Y|X}(i|x)$
        - 2) $i^*(x) = \text{argmax}_i[P_{X|Y}(x|i)P_Y(i)]$
        - 3) $i^*(x) = \text{argmax}_i[\text{log} P_{X|Y}(x|i) + \text{log}P_Y(i)]$
        
        By introducing a modle for the class-conditional distributions, we can express this as a simple equation
        - e.g. for the multivariate Gaussian
        $$
        P_{X|Y}(x|i) = \frac{1}{\sqrt{(2\pi)^d |\Sigma_i|}}\text{exp}\{-\frac{1}{2}(x-\mu_i)^T \Sigma_i^{-1} (x-\mu_i)\}
        $$
        
        ### Geometric Interpretation
        
        For Gaussian classes equal but arbitrary covariance
        
        $$
        \boxed{
        \begin{array}{l}
        w = \Sigma^{-1}(\mu_i - \mu_j)\\
        x_0 = \frac{\mu_i + \mu_j}{2} - \frac{1}{(\mu_i - \mu_j)^T\Sigma^{-1}(\mu_i - \mu_j)}\text{log}\frac{P_Y(i)}{P_Y(j)}(\mu_i - \mu_j)
        \end{array}
        }
        $$
        
        ### Implementation
        
        We do have an optimal solution, but in practice we do not know the values of the parameters $\mu, \Sigma, P_Y$
        - We have to somehow estimate these values
        - This is Ok, we can come up with witha n estimate from a training set
        - e.g. use the average value as an estimate for the mean
        $$
        \boxed{
        \begin{array}{l}
        w = \Sigma^{-1}(\hat{\mu_i} - \hat{\mu_j})\\
        x_0 = \frac{\hat{\mu_i} + \hat{\mu_j}}{2} - \frac{1}{(\hat{\mu_i} - \hat{\mu_j})^T\Sigma^{-1}(\hat{\mu_i} - \hat{\mu_j})}\text{log}\frac{P_Y(i)}{P_Y(j)}(\hat{\mu_i} - \hat{\mu_j})
        \end{array}
        }
        $$
        
        Sources of error:
        - Parameter estimation
        - Model Error: We assume the model is Gaussian, but that may not be the case.
                    
        ### Important
        - At this point all optimiality calims for the BDR cease to be valid
        - The BDR is guaranteed to achieve the minimum loss when we use the true probabilities
        - When we plug in the probability estiamtes, we could be implementing a classifier that is quite distant from the opitmal
            - e.g. if the $P_{X|Y}(x|i)$ looks non-Gaussian
            - We could never approximate them well using Gaussian parameters
            
        
        """)
        
    if l7_sub == "Maximum Likelihood Estimation":
        st.markdown(r"""
        
        The sample mean is taken to be the Maximum Likelihood Estimator under the assumption that the distribution is Gaussian
        
        $$
        \hat{\mu} = \frac{1}{n} \sum_i x_i
        $$
        
        ### Maximum Likelihood
        
        This has three steps:
        1) We choose a parametric model for all probabilities
            - To make this clear, we denote the vector of parameters by $\Theta$ and the class-conditional distributions by
            $$
            P_{X|Y}(x|i; \Theta)
            $$
            - Note that this means that $\Theta$ is NOT a random variable (otherwise it would have to show up as a subscript)
            - It is simply a parameter, and the probabilities are a function of this parameter
            
        2) We assemble a collection of datasets
            - $D^{(i)} = \{x_1^{(i)}, \dots, x_n^{(i)}\}$ set of examples drawn independently from class $i$
            
        3) We select parameters of the class $i$ to be the ones that maximize the probaility of the data from that class
            $$
            \begin{array}{l}
            \Theta_i = \text{argmax}_{\Theta}P_X(D:\Theta)\\
                = \text{argmax}_{\Theta}\text{log}P_X(D;\Theta)
            \end{array}
            $$
            - The function $P_X(D:\Theta)$ is called the likelihood of the parameter $\Theta$ with respect to the data or simply the likelihood function
            
        **PDFs vs Likelihood functions**
        - PDF: function of x
        - Likelihood function: x is known so it is a function of $\mu, \sigma^2$ "a 2D function where Likelihood is the height"
        
        - Note that the likelihood function is a function of the parameters
        - It does not have the same shape as the density itself
        - e.g. the likelihood function of a Gaussian is not bell-shaped
        - The likelihood is defined only after we have a sample
        $$
        \boxed{
        P_X(d; \Theta) = \frac{1}{\sqrt{(2\pi)\sigma^2}}\text{exp}\{-\frac{(d-\mu)^2}{2\sigma^2}\}
        }
        $$
        
        - Given a sample, to obtain ML estimat we need to solve
        $$
        \boxed{
        \Theta^* = \text{argmax}_\Theta P_X(D; \Theta)
        }
        $$
        - when $\Theta$ is a scalar, this is high-school calculus
        - we have a maximum when
            - First derivative is zero
            - Second derivative is negative
        ### The Gradient
        - in higher dimensions, the generalization of the derivative is the gradient
        - the gradient fo a function $f(x)$ at $x$ is
        $$
        \nabla f(x) = \left(\frac{\partial{f}}{\partial{x_0}}(x), \dots, \frac{\partial{f}}{\partial{x_{n-1}}}(x)\right)^T
        $$
        - The gradient has a nice geometric interpretation
            - It points in the direction of maximum growth of the function
            - Which makes it perpendicular to the contours where the function is constant
            
        Note that if $\nabla f = 0$
        - There is no direction of growth
        - Also $-\nabla f = 0$, and there is no direction of decrease
        - We are either at a local minimum or maximum or "saddle point"
        
        Conversely, at local min or max or saddle point
        - no direction of growth or decrease
        - $\nabla f = 0$
        
        This shows taht we have a critical point and if only if $\nabla f = 0$
        
        To determine which type we need second order conditions
        
        ### The Hessian
        
        The extension of the second-order derivative is the Hessian Matrix
        $$
        \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_0^2}(x) & \dots & \frac{\partial^2 f}{\partial x_0 \partial x_{n-1}} (x) \\ & \vdots & \\ 
        \frac{\partial^2 f}{\partial x_{n-1}\partial x_0}(x) & \dots & \frac{\partial^2 f}{\partial x^2_{n-1}}\end{bmatrix}
        $$
        
        At each point $x$, gives us the quadratic function
        $$
        x^t \nabla^2 f(x) x
        $$
        that best approximates $f(x)$
        
        This means that when gradient is zero at $x$, we have
        - a maximum when function can be approximated by an "upward facine" quadratic
        - a minimum when function can be approximated by a "downward facing" quadratic
        - a saddle point otherwise
        
        For any matrix $M$, the function
        $$
        x^t Mx
        $$
        is
        - upwards facing quadratic when M is negative definite
        - downwards facing quadratic when M is positive definite
        - saddle otherwise
        
        hence, all that matters is the positive definiteness of the Hessian
        
        we have a maximum when Hessian is negative definite
        
        ### Maximum Likelihood
        
        In summary, given a sample, we need to solve
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D; \Theta)
        $$
        
        The solutions are the parameters such that
        $$
        \nabla_\Theta P_X (D;\Theta) = 0
        $$
        $$
        \theta^t \nabla_\Theta^2 P_X (D;\Theta) \theta \leq 0, \quad \theta \in R^n
        $$
        Note that you always have to check the second order condition
        
        Lets consider the Gaussian example
        $$
        \boxed{
            f(T) = \frac{1}{\sqrt{\sigma_T \sqrt{2\pi}}}e^{-\frac{1}{2}(\frac{T-\hat{T}}{\sigma_T})^2}
        }
        $$
        Given a sample $T_1, \dots, T_n$ of independent points
        
        The likelihood is
        
        $$
        \boxed{
        L(T_1, T_2, \dots, T_N | \bar{T}, \sigma_T) = L = \prod_{i=1}^N [\frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{T-bar{T}}{\sigma_T})^2]}]\\
        L = \frac{1}{\sigma_T \sqrt{2\pi}^N} e^{-\frac{1}{2}\sum_{i=1}^N\left(\frac{T_i - \bar{T}}{\sigma_T}\right)^2}
        }
        $$
        
        And the log-likelihood is
        $$
        \boxed{
        \Lambda = ln L = -\frac{N}{2}ln(2\pi) - N ln \sigma_T - \frac{1}{2} \sum_{i=1}^N \left(\frac{T_i - \bar{T}}{\sigma_T}\right)^2
        }        
        $$
        The derivative with respect to the mean is zero when
        $$
        \frac{\partial (\Lambda)}{\partial T} = \frac{1}{\sigma_T^2}\sum_{i=1}^N (T_i - \bar{T}) = 0
        $$
        or
        $$
        \bar{T} = \frac{1}{N}\sum_{i=1}^N T_i
        $$
        Note that this is just the sample mean
        
        The derivative with respect to the variance is zero when
        $$
        \frac{\partial (\Lambda)}{\partial \sigma_T} = -\frac{N}{\sigma_T} + \frac{1}{\sigma^3}\sum_{i=1}^N(T_i - \bar{T})^2 = 0
        $$
        or
        $$
        \boxed{
            \sigma_T^2 = \frac{1}{N}\sum+{i=1}^N (T_i - \bar{T})^2
        }
        $$
        Note that this is just the sample variance
        
        ### Homework
        
        Show that the Hessian is negative definite
        $$
        \boxed{
            \theta^2 \nabla_\Theta^2 P_X (D; \Theta) \theta \leq 0, \quad \theta \in R^n
        }
        $$
        
        Show that these formulas can be generalized to the vector case
        - $D = {x_1, \dots, x_n}$ set of examples from class i
        - The ML estimates are
        $$
        \mu_i = \frac{1}{n}\sum_j x_j^{(i)}, \quad \Sigma_i = \frac{1}{n}\sum_j (x_j^{(i)} - \mu_i)(x_j^{(i)} - \mu_i)^T
        $$
        Note that the ML solution is usually intuitive
        """)
        
if section == "Bias and Variance":

    st.header("Lecture 8 - Wednesday 10/22/25")
    
    l8_sub = st.radio(
        "Topics",
        ["Maximum Likelihood Estimation", "Bias and Variance"]
    )
    if l8_sub == "Maximum Likelihood Estimation":
        st.markdown(r"""
        
        #### Parameter Estimation in three steps   
        1) Choose a parametric model for probabilities, to make this clear we denote the vector parameters by $\Theta$
        $$
        P_X(x; \Theta)
        $$
        note this means that $\Theta$ not a random variable
        
        2) Assemble $D = \{X_1, \dots, X_n\}$ of examples drawn independently
        
        3) Select the parameters that maximize the probability of the data
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D;\Theta)\\
        = \text{argmax}_\Theta \text{log} P_X(D;\Theta)
        $$
        
        $P_X(D;\Theta)$ is the likelihood parameter $\Theta$ with respect to the data
        
        #### In summary
        
        Given a sample we need to solve
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D;\Theta)
        $$
        
        The solutions are the parameters sucha that
        $$
        \nabla_\Theta P_X(X;\Theta) = 0\\
        \theta^t \nabla_\Theta^2 P_X(x;\theta)\theta \leq 0, \quad \forall \theta \in R^n
        $$
        
        Note that you always have to check the second order condition
        
        #### Gaussian Case
        $$
        f(T) = \frac{1}{\sigma_T \sqrt{2\pi}}e^{-\frac{1}{2} \left(\frac{T- \bar{T}}{\sigma_T}\right)^2}
        $$
        
        Given a sample $T_1, \dots, T_N$ of independent points, 
        
        The log likelihood is
        $$
        \Lambda = ln L = -\frac{N}{2}ln(2\pi) - N ln \sigma_T - \frac{1}{2} \sum_{i=1}^N \left(\frac{T_i - \bar{T}}{\sigma_T}\right)^2
        $$
        
        The ML estimates of the mean and variance are
        
        $$
        \bar{T} = \frac{1}{N} \sum_{i = 1}^N T_i \quad \hat{\sigma}_T^2 = \frac{1}{N} \sum_{i = 1}^N(T_i - \bar{T})^2
        $$
        
        ### Estimators
        When we talk about estimators, it is important to keep in mind that
        - an estimate is a number
        - an estimator is a random variable
        $$
        \hat{\theta} = f(X_1, \dots, X_n)
        $$
        
        An estimate is the value of the estimator for a given sample
        
        If $D = \{X_1, \dots, X_n\}$, when we say $\hat{\mu} = \frac{1}{n} \sum_j x_j$
        
        what we mean is $\hat{\mu} = f(X_1, \dots, X_n)|_{X_1 = x_1, \dots, X_n = x_n}$ with $f(X_1, \dots, X_n) = \frac{1}{n} \sum_j X_j$ with $X_i$ as the random varialbes
        
        """)
        
    if l8_sub == "Bias and Variance":
        st.markdown(r"""
        ### Bias and Variance
        
        - We know how to produce estimators by ML
        - how do we evaluate an esitmator?
        - $Q_1$: is the expected value equal to the true value?
        - This is measured by the bias
        $$
        \hat{\theta} = f(X_1, \dots, X_n)\\
        Bias(\hat{\theta}) = E_{X_1, \dots, X_n}[f(X_1, \dots, X_n)-\theta]
        $$      
        
        - an estimator that has bias will usually not coverge tot he perfect estimate $\theta$ no matter how large the sample is
        - E.g. if $\theta$ is negative and the estimator is $f(X_1, \dots, X_n) = \frac{1}{n} \sum_j X_j^2$, the bias is clearly non-zero
        
        The estimators is said to be biased
        - This means that it is not expressive enough to approximate the true value arbitrarily well
        - This will be clearer when we talk about density estimation
        $Q_2$: assuming that the estimator converges to the true value, how many sample points do we need?
        - This can be measured by the variance
        $$
        Var(\hat{\theta}) = E_{X_1, \dots, X_n}\{f(X_1, \dots, X_n) - E_{X_1, \dots, X_n}[f(X_1, \dots, X_n)]^2\}
        $$
        
        #### Example (Bias)
        - ML estimator for the mean of a Gaussian $N(\mu, \sigma^2)$
        $$
        \begin{array}{l}
        Bias(\hat{\mu}) = E_{X_1, \dots, X_n}[\hat{\mu}-\mu] = E_{X_1, \dots, X_n}[\hat{\mu}]- \mu \\
        = E_{{X_1, \dots, X_n}}[\frac{1}{n}\sum_i X_i] - \mu\\
        = \frac{1}{n} \sum_i E_{X_1, \dots, X_n}[X_i] - \mu\\
        = \frac{1}{n}\sum_i E_{X_i}[X_i] - \mu\\
        = \mu - \mu = 0
        \end{array}
        $$
        
        **Linearity of Expectation**
        $$
        \begin{array}{l}
        E_x[af(x) + bg(x)] = \int P_x(x)\{af(x) + bg(x)\}dx\\
        = a \in P_x(x)f(x) + b \int P_x(x) g(x) dx \\
        = a E[f(x)] + b E_x[g(x)]
        \end{array}
        $$
        
        $$
        \begin{array}{l}
        Var(\hat{\mu}) = E_{X_1, \dots, X_n}\{(\hat{\mu} - E_{X_1, \dots, X_n}[\hat{\mu}]_)^2\} = E_{X_1, \dots, X_n}\{(\hat{\mu} - \mu)^2\}\\
        = E_{X_1, \dots, X_n}\{(\frac{1}{n} \sum_i X_i \mu)^2\}\\
        = \frac{1}{n^2} E_{X_1, \dots, X_n}\{(\sum_i (X_i - \mu))^2\}\\
        = \frac{1}{n^2}E_{X_1, \dots, X_n}\{\sum_{ij}(X_i - \mu)(X_j - \mu)\}
        \end{array}
        $$
        
        The ML estimator for the mean of a Gaussian
        $$
        Var(\hat{\mu}) = \frac{1}{n^2} \sum_{ij}E_{X_i, X_j}[(X_i - \mu)(X_j - \mu)] = \frac{1}{n^2}\sum_{ij}\sigma_{ij}
        $$
        And since $X_i, X_j$ are independent, $\sigma_{ij} = 0, \quad \forall i \neq j$
        $$
        Var(\hat{\mu}) = \frac{1}{n^2} \sum_i \sigma_i^2 = \frac{\sigma^2}{n}
        $$
        
        The variance goes to zero as $n$ increases
        
        #### In summary
        
        For ML estimator for the mean of a Gaussian $N(\mu, \sigma^2)$
        $$
        E[\hat{\mu}] = \mu \quad Var(\hat{\mu}) = \frac{\sigma^2}{n}
        $$
        
        This means that if I have a large sample, the value of the estimate will be close to the true value with high probability
        
        #### Example (Variance)
        The ML estimator for the variance of a Gaussian is a biased estimator
        $$
        \hat{\sigma}^2 = \frac{1}{n}\sum_i (X_i - \hat{\mu})^2 = \frac{1}{n}\sum_i (X_i^2 - 2X_i \hat{\mu} + \hat{\mu}^2)\\
        = \frac{1}{n} \sum_i X_i^2 - \hat{\mu}^2
        $$
        
        The expected value is 
        $$
        E_{X_1, \dots, X_n}[\hat{\sigma}^2] = \frac{1}{n} \sum_i E_{X_1, \dots, X_n}[X_i^2] - E_{X_1, \dots, X_n}[\hat{\mu}^2]\\
        = \frac{1}{n} \sum_i E_{X_i}[X_i^2] - E_{X_1, \dots, X_n}[\hat{\mu}^2] = E_X[X^2] - E_{X_1, \dots, X_n}[\hat{\mu}^2]
        $$
        
        Using
        $$
        E_{X_1, \dots, X_n}[\hat{\mu}^2] = E_{X_1, \dots, X_n}\left[\frac{1}{n^2} \sum_{ij} X_i X_j\right] = \frac{1}{n^2}\sum_{ij}E_{X_i, X_j}[X_i, X_j]\\
        = \frac{1}{n^2}\sum_i E_{X_i}[X_i^2] + \frac{1}{n^2} \sum_{i, j \neq i} E_{X_i, X_j}[X_i, X_j]\\
        = \frac{1}{n} E_X [X^2] + \frac{1}{n^2} \sum_{i, j \neq i} E_{X_i} [X_i]E_{X_j}[X_j]\\
        = \frac{1}{n} E_X[X^2] + \frac{1}{n^2} \sum_i E_{X_i}[X_i](n-1)E_X[X]
        $$
        
        Using
        $$
        E_{X_1, \dots, X_n}[\hat{\mu}^2] = \frac{1}{n}E_X[X^2] + \frac{1}{n^2}\sum_i E_{X_i}[X_i](n-1)E_X[X]\\
        = \frac{1}{n} E_X [X^2] + \frac{(n-1)}{n}(E_X[X])^2\\
        = \frac{1}{n} E_X [X^2] + \frac{(n-1)}{n} \mu^2
        $$
        
        We get
        $$
        E_{X_1, \dots, X_n}[\hat{\sigma}^2] = E_X[X^2] - E_{X_1, \dots, X_n}[\hat{\mu}]\\
        = \frac{(n-1)}{n}E_X[X^2]- \frac{(n-1)}{n}\hat{\mu} = (1 - \frac{1}{n})\sigma^2
        $$
        
        This is a biased estimator, when $n$ is large, the bias goes away
        
        If $n$ is small, the variance is already large due to the Expected Value
        
        
        ### Important Note
        
        Since the estimator is a random variable
        - We can never say that an estimate is obtained with more samples is "better" than an estimate from less samples
        - e.g. if
        $$
        \mu_1 = \frac{1}{100} \sum_{i=1}^{100} X_i \quad \mu = \frac{1}{10000} \sum_{i=1}^{10000} X_i
        $$
        we measure and obtain
        $$
        \hat{\mu}_1 = 10.5 \quad \hat{\mu}_2 = 10.3
        $$
        
        - We can never know, all we know is that
        $$
        \mu_1 = N(\mu, \sigma^2/100) \quad \mu_2 = N(\mu, \sigma^2/10,000)
        $$
        
        The estimate is only a sample, estimators are random variables.
        """)

    st.divider()
    st.header("Lecture 9 - Monday 10/27/25 ")  
    l9_sub = st.radio(
        "Topics",
        ["Review", "Bias and Variance"]
    )     
    if l9_sub == "Review":
        st.markdown(r"""
        ### Maximum Likelihood Estimation
        
        #### Parameter Estimation in three steps   
        1) Choose a parametric model for probabilities, to make this clear we denote the vector parameters by $\Theta$
        $$
        P_X(x; \Theta)
        $$
        note this means that $\Theta$ not a random variable
        
        2) Assemble $D = \{X_1, \dots, X_n\}$ of examples drawn independently
        
        3) Select the parameters that maximize the probability of the data
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D;\Theta)\\
        = \text{argmax}_\Theta \text{log} P_X(D;\Theta)
        $$
        
        $P_X(D;\Theta)$ is the likelihood parameter $\Theta$ with respect to the data
        
        #### In summary
        
        Given a sample we need to solve
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D;\Theta)
        $$
        
        The solutions are the parameters sucha that
        $$
        \nabla_\Theta P_X(X;\Theta) = 0\\
        \theta^t \nabla_\Theta^2 P_X(x;\theta)\theta \leq 0, \quad \forall \theta \in R^n
        $$
        
        Note that you always have to check the second order condition
        
        #### Gaussian Case
        $$
        f(T) = \frac{1}{\sigma_T \sqrt{2\pi}}e^{-\frac{1}{2} \left(\frac{T- \bar{T}}{\sigma_T}\right)^2}
        $$
        
        Given a sample $T_1, \dots, T_N$ of independent points, 
        
        The log likelihood is
        $$
        \Lambda = ln L = -\frac{N}{2}ln(2\pi) - N ln \sigma_T - \frac{1}{2} \sum_{i=1}^N \left(\frac{T_i - \bar{T}}{\sigma_T}\right)^2
        $$
        
        The ML estimates of the mean and variance are
        
        $$
        \bar{T} = \frac{1}{N} \sum_{i = 1}^N T_i \quad \hat{\sigma}_T^2 = \frac{1}{N} \sum_{i = 1}^N(T_i - \bar{T})^2
        $$
        
        ---
        
        ### Estimators
        When we talk about estimators, it is important to keep in mind that
        - an estimate is a number
        - an estimator is a random variable
        $$
        \hat{\theta} = f(X_1, \dots, X_n)
        $$
        
        An estimate is the value of the estimator for a given sample
        
        If $D = \{X_1, \dots, X_n\}$, when we say $\hat{\mu} = \frac{1}{n} \sum_j x_j$
        
        what we mean is $\hat{\mu} = f(X_1, \dots, X_n)|_{X_1 = x_1, \dots, X_n = x_n}$ with $f(X_1, \dots, X_n) = \frac{1}{n} \sum_j X_j$ with $X_i$ as the random varialbes
        
        ---
        
        ### Bias and Variance
        
        - We know how to produce estimators by ML
        - how do we evaluate an estimator?
        
        **$Q_1$: is the expected value equal to the true value?**
        - This is measured by the bias
        $$
        \hat{\theta} = f(X_1, \dots, X_n)\\
        Bias(\hat{\theta}) = E_{X_1, \dots, X_n}[f(X_1, \dots, X_n)-\theta]
        $$      
        
        - an estimator that has bias will usually not coverge tot he perfect estimate $\theta$ no matter how large the sample is
        - E.g. if $\theta$ is negative and the estimator is $f(X_1, \dots, X_n) = \frac{1}{n} \sum_j X_j^2$, the bias is clearly non-zero
        
        The estimators is said to be biased
        - This means that it is not expressive enough to approximate the true value arbitrarily well
        - This will be clearer when we talk about density estimation
        
        **$Q_2$: assuming that the estimator converges to the true value, how many sample points do we need?**
        - This can be measured by the variance
        $$
        Var(\hat{\theta}) = E_{X_1, \dots, X_n}\{f(X_1, \dots, X_n) - E_{X_1, \dots, X_n}[f(X_1, \dots, X_n)]^2\}
        $$
        The variance usually decreases as one collects more training samples
        
        ### Important Note
        
        Since the estimator is a random variable
        - We can never say that an estimate is obtained with more samples is "better" than an estimate from less samples
        - e.g. if
        $$
        \mu_1 = \frac{1}{100} \sum_{i=1}^{100} X_i \quad \mu = \frac{1}{10000} \sum_{i=1}^{10000} X_i
        $$
        we measure and obtain
        $$
        \hat{\mu}_1 = 10.5 \quad \hat{\mu}_2 = 10.3
        $$
        
        - We can never know, all we know is that
        $$
        \mu_1 = N(\mu, \sigma^2/100) \quad \mu_2 = N(\mu, \sigma^2/10,000)
        $$
        
        The estimate is only a sample, estimators are random variables.
        
        We use the estimators to compute
        $$
        P(|\mu_2 - \mu| < |\mu_1 - \mu|)
        $$
        
        but there is always a probability that the estimate produced by $\mu_1$ is better than that produced by $\mu_2$
        - even though $\mu_2$ has a much smaller variance
        - all that we can hope for, is to make the estimator better in a probabilistic sense
        - this means making
        $$
        P_{\hat{\Theta}}(\theta)
        $$
        as concentrated as possible around the true value
        - In this sense, emphasizing the bias or variance can be wrong
        """)
        
    if l9_sub == "Bias and Variance":
        st.markdown(r"""
        We really care about the conjunction of the two factors
        - Working hard to decrease variance is bias is large is useless
        - Working hard to decrease bias is variance is large is useless          
        
        ### Mean Square Error
        One possibility to account for both bias and variance is to minimize the mean squared error
        - if $\hat{\theta} = f(X_1, \dots, X_n)$
        - Then: $MSE(\hat{\theta}) = E_{X_1, \dots, X_n}[\{f(X_1, \dots, X_n) - \theta\}^2]$
        
        The connection to bias and variance follows from
        $$
        MSE(\hat{\theta}) = E[\{\hat{\Theta} - E[\hat{\Theta}] + E[\hat{\Theta}] - \theta\}^2]\\
            = E[\{\hat{\Theta} - E[\hat{\Theta}]\}^2] + 2E[\{\hat{\Theta} - E[\hat{\Theta}]\}\{E[\hat{\Theta}] -\theta\}] + E[\{E[\hat{\Theta}] -\theta\}^2]
        $$
        
        $$
        \begin{array}{l}
        = var(\hat{\Theta}) + 2E(\hat{\Theta} - E[\hat{\Theta}]\})\{E[\hat{\Theta}] -\theta\} + \{E[\hat{\Theta}] - \theta\}^2\\
        = var(\hat{\Theta}) + 2E(\hat{\Theta} - E[\hat{\Theta}]\})\{E[\hat{\Theta}] -\theta\} + Bias^2(\hat{\Theta})
        \end{array}
        $$
        
        And
        $$
        MSE(\hat{\Theta}) = var(\hat{\Theta}) + Bias^2(\hat{\Theta})
        $$
        
        ### Bias Variance Tradeoff
        
        In general, the MSE estimator has non-zero bias and vairnace
        
        We can only reduce bias at the cose of increased variance and vice versa
        - Suppose we are not happy with the $1/n$ decay of the variance of
        $$
        \hat{\mu} = \frac{1}{n} \sum_i X_i
        $$
        - One possibility is to use
        $$
        \hat{\mu} = \frac{\alpha}{n} \sum_i X_i = \alpha \hat{\mu}
        $$
        
        - This has
        $$
        E[\hat{\mu}] = \alpha \mu \quad Bias[\hat{\mu}] = (1-\alpha)\mu \quad var[\hat{\mu}] = \frac{\alpha^2 \sigma^2}{n}
        $$
        
        By choosing $\alpha < 1$ we can decrease the variance ,but the bias will no longer be zero
        - What value of $\alpha$ minimizes the MSE?
        $$
        MSE[\hat{\mu}] = var[\hat{\mu}] + Bias^2[\hat{\mu}]\\
            = \frac{\alpha^2 \sigma^2}{n} + (1-\alpha)^2 \mu^2
        $$
        
        And
        $$
        \frac{\partial MSE[\hat{\mu}]}{\partial \alpha} = 2\alpha \frac{\sigma^2}{n} - 2(1-\alpha)\mu^2
        $$
        
        - From which 
        $$
        \frac{\partial MSE[\hat{\mu}]}{\partial \alpha} = 0 \iff \alpha \frac{\sigma^2}{n} + \alpha \mu^2 = \mu^2 \\
        \iff \alpha(\frac{\sigma^2}{n} + \mu^2) = \mu^2 \iff \alpha = \frac{\mu^2}{\frac{\sigma^2}{n} + \mu^2}
        $$
        
        And the MSE estimator of $\mu$ is
        $$
        \hat{\mu} = \frac{\mu^2}{\sigma^2 + n \mu^2}\sum_i X_i
        $$
        
        one can immediately detect a problem
        - The optimal estimator depends on the quantity that we are trying to estimate
        - The estimator is unrealizable
        
        ### Estimators
        
        Unrealizable solutions are a common source of problems for the MSE estimator
        
        One alternative is to
        - Constrain the estimator to be in a class (e.g. unbiased)
        - Find, among all solutions in the class, that of the least MSE
        
        Many ideas on how to do this
        - BLUE: best linear unbiased estimator
        - MVUE: minimum variance unbiased
        - check the parameter estimation literature
        
        WHy is the ML estimator so popular?
        - Many of these alternatives are frequenly unrealizable
        - The ML solution typically makes intuitive sense
        - Connections to Bayesian estimation (we will talk about this later)
        
        Consider BLUE estimators for the population mean
        $$
        \mu_{BLUE} = \sum_i w_i X_i
        $$
        
        - What are the weights $w_i$ such that 
        $$
        E[\mu_{BLUE}] = E[X] = \mu\\
        var[\mu_{BLUE}] = MSE[X]
        $$
        is minimal?
        
        - The answer is
        $$
        \mu_{BLUE} = \frac{1}{n} \sum_i X_i
        $$
        
        - Note that this holds independently of whether $X$ is Gaussian
        - But for Gaussian X, it is the same as ML
        - When there is an easy realizable solution, ML gets it
        
        ### Least Squares
        There are also interesting connections between ML estimation and least squares methods
        
        E.G. in a regression problem we have
        - two random variables $X$ and $Y$
        - a dataset of examples $D = \{(x_1, y_1), \dots, (x_n, y_n)\}$
        - a parametric model of the form
        $$
        y = f(x;\Theta) + \epsilon
        $$
        - where $\Theta$ is a parameter vector, and $\epsilon$ a random variable that accounts for noise
        - e.g. $\epsilon ~ N(0, \sigma^2)$
        
        Assuming that the family of models is known, e.g.
        $$
        f(x;\Theta) = \sum_{i=0}^K \theta_i x^i
        $$
        - This is really just a problem of parameter estimation where the data is distributed as
        $$
        P_{Y|X}(y|x;\theta) = G(y, f(x;\theta), \sigma^2)
        $$
        - note that $X$ is always known, and the mean is a function of $x$ and $\Theta$
        - in the homework, you will show that ML estimate is
        $$
        \Theta^* = [\Gamma^T \Gamma]^{-1} \Gamma^T y
        $$
        
        where
        $$
        \Gamma = \begin{bmatrix}1 & \dots & x_i^K \\ & \vdots & \\ 1 & \dots & x_n^K\end{bmatrix}
        $$
        
        Conclusion
        - Least squares estimation is just ML estimation under the assumption of 
            - Gaussian Noise
            - independent sample
            - $\epsilon ~ N(0, \sigma^2)$
        - Once again, probability makes the assumptions explicit
        $$
        \sum_i log P_{Y|X}(y_i |x_i, \theta)\\
        \sum log[\alpha e^{-(\frac{y-x_i}{2\sigma^2})^2}]
        $$
        
        $$
        \Theta^* = \text{min}_\theta \sum_i (f(x_i, \theta) - y_i)^2
        $$
        
        Due to the connection to parameter estimation, we can also talk about the quality of the least squares solution
        In particular we know that
        - It is unbiased
        - Variance goes to zero as the number of points increase
        - It is the BLUE esitmator for $f(x, \theta)$
            
        Under the statistical formulation we can also see how the optimal estimator changes with assumptions
        
        ML estimation can also lead to 
        - weighted least squares
        - minimzation of $L_p$ norms
        - robust estimators
        """)
     
if section == "Bayesian Parameter Estimation":
    st.header("Lecture 10 - Wednesday 10/29/25")
    l10_sub = st.radio(
        "Topics",
        ["Review", "Bayesian Parameter Estimation", "Bayes vs Maximum Likelihood"]
    )
    if l10_sub == "Review":
        st.markdown(r"""
        ### Maximum Likelihood
        
        #### Parameter Estimation in three steps   
        1) Choose a parametric model for probabilities, to make this clear we denote the vector parameters by $\Theta$
        $$
        P_X(x; \Theta)
        $$
        note this means that $\Theta$ not a random variable
        
        2) Assemble $D = \{X_1, \dots, X_n\}$ of examples drawn independently
        
        3) Select the parameters that maximize the probability of the data
        $$
        \Theta^* = \text{argmax}_\Theta P_X(D;\Theta)\\
        = \text{argmax}_\Theta \text{log} P_X(D;\Theta)
        $$
        
        $P_X(D;\Theta)$ is the likelihood parameter $\Theta$ with respect to the data          
        
        ### Estimators
        
        ML Estimator for the mean of a Gaussian
        $$
        E[\hat{\mu}] = \mu \quad Var(\hat{\mu}) = \frac{\sigma^2}{n}
        $$
        
        This means that if I have a large sample, the value of the estimate will be close to the true value with high probability
        """)
        
    if l10_sub == "Bayesian Parameter Estimation":
        st.markdown(r"""
        ### Bayesian Parameter Estimation
        
        Bayesian parameter estimation is an alternative framework for **parameter estimation**
        - It turns out that the division between Bayesian and ML methods is quite fundamental
        
        It stems from a different way of interpreting probabilities
        - Frequentist vs. Bayesian
        
        There is a long debate about which is best
        - This debate goes to the core of what probabilities mean
        
        To understand it, we have to distinguish two components
        - The definition of probability(this does not change)
        - The assessment of probability (this changes)
        
        Lets start with a brief review of the part that does not change
        
        Note:
        - Maximum Likelihood Estimation comes from the Frequentist School
        - Bayesian Parameter Estimation comes from the Bayesian School
        ---
        #### Probability
        Probability is a language to deal with processes that are non-deterministic
        
        Examples:
        - If i flip a coin 100 times, how many times can I expect to see heads?
        - What is the weather going to be like tomorrow?
        - Are my stocks going to be up or down?
        - Am I in front of a classroom or is this just a picture of it?
        ---
        #### Sample Space
        The most important concept is that of a sample space
        
        Our process defines a set of events
        - These are the outcomes or states of the process
        
        Example:
        - We roll a pair of dice
        - Call the value on the up face at the nth toss of $x_n$
        - Note that posible events such as
            - Odd number on second throw
            - Two sixes
            - $x_1 = 2$ and $x_2 = 6$
        - Can all be expressed as combinaitions of the sample space events
        
        Is the list of possible events that satisfies the following properties
        - Finest grain: all possible distinguishable events are listed separaetly
        - Mutually exclusive: if one event happens, the other does not (if $x_1 = 5$ it cannot be anything else)
        - Collectively exhaustive: any possible outcome can be expressed as unions of sample space events
        
        Mutually exclusive property simplifies the calculation of the probability of complex events
        
        Collectively exhaustive means that there is no possible outcome to which we cannot assign a probability
        
        ---
        #### Probability Measure
        Probability of an event
        - number expressing the chance that the event will be the outcome of the process
        
        Probability measure: satisfies three axioms
        - $P(A) \geq 0$ for any event $A4
        - $P(\text{universal event}) = 1$
        - if $A$ union $B$ is null, then $P(A+B) = P(A) + P(B)$
        
        All of this
        - has to do with the definition of probability
        - Is the same under Bayes and frequentist views
        
        What changes is how probabilities are assessed
        
        ---
        #### Frequentist View
        
        Under the frequentist view, probabilities are relative frequencies
        - I throw my dice $n$ itmes
        - In $m$ of those the sum is 5
        - I say that
        $$
        P(sum = 5) = \frac{m}{n}
        $$
        
        This is intimately connected with the ML method
        - It is the ML estimate for the probability of a Bernoulli process with states ("5", "everything else")
        - Makes sense when we have a lot of observations
            - No bias, decreasing vairance, converges to true probability
            
        #### Problems
        Many instances where we do not have a large number of observations
        
        COnsider the problem of crossing a street
        
        This is a decision problem with two states
        - $Y = 0$: I am going to get hurt
        - $Y = 1$: I will make it safely
        
        Optimal decision computable by Bayes decision rule
        - Collect some measurements that are informative
        - e.g. (X = {size, distance, speed}) of incoming cars)
        - Collect examples under both states and estimate all probabilities
        
        Somehow this does not sound like a great idea!
        
        Under frequentist view
        - You need to repeat an experiment a large number of times to estimate any probabilities
        
        Yet people are very good at 
        - Estimating probabilities for problems in which it is impossible ot set up such experiments
        
        For example:
        - Will I die if I join the army?
        - Will Democrats or Republicans win the next election?
        - Is there a God?
        - Will I graduate in two years?
        
        To the point where they make life-changin decisions based on these probability estimates
        
        ---
        #### Subjective Probability
        
        This motivates an alternative definition of probabilites
        - Note that this has to do more with how probabilities are assessed than with the probability definition itself
        - We still have a sample space, a probability measure etc
        - However, the probabilites are not equated to relative counts
        
        This is usually referred to as subjective probability
        
        Probabilities are degrees of belief on the outcomes of the experiment
        - They are individual (vary from person to person)
        - They are not ratios of experimental outcomes
        
        e.g.
        - For very religious person $P(\text{God exists}) ~ 1$
        - For casual churchgoer $P(\text{God exists}) ~ 0.8$
        - For non-religious $P(\text{God exists}) ~ 0$
        
        #### Problem
        In pracitce why do we cdare about this?
        
        Under the notion of subjective probability, the entire ML framework makes little sense
        - There is a magic number that is estimated from the world and determines our beliefs
        - To evaluate my estimates I have to run expereiments over and over and measure quanitites like bias and varaince
        - This is not how people behave, when we make estiamtes, we attach a degree of confidence to them, without further experiments
        - There is only one model (the ML model) for the probability of the data, no multiple explanations
        - There is no way to specify that some models are, apriori ,better than others
        
        """)
    
    if l10_sub == "Bayes vs Maximum Likelihood":
        st.markdown(r"""
        ### Bayes vs Maximum Likelihood
        
        The main difference with respect to ML is that in the Bayesian case $\Theta$ is a random variable
        
        basic concepts
        - Training set of examples drawn independently
        - Probability density for observations given parameter
        $$
        P_{X|\Theta}(x|\theta)
        $$
        
        - prior distribution for parameter configurations
        $$
        P_\Theta(\theta)
        $$
        
        that encodes prior beliefs about them
        
        Goal: to compute the posterior distribution
        $$
        P_{\Theta| X}(\theta| D)
        $$   
        
        There are a number of significant differences between Bayesian and ML estimates
        
        $D_1$:
        - Maximum Likelihood produces a number, the best estimate
        - To measure its goodness, we need to measure bias and variance
        - This can only be done with repeated experiments
        - Bayes produces a complete characterization of the parameter from a single dataset
        - In addition to the most probably estimate, we obtain a characterization of the uncertainty
        
        $D_2$: optimal estimate
        - Under ML there is one best estimate
        - Under Bayes there is no best estimate
        - Only a random variable that takes different values with different probabilities
        - Technically speaking, it makes no sense to talk abou the best estimate
        
        $D_3$: predictions
        - remember that we do not really care about the parameters themselves
        - They are needed only in the sense that they allow us to build models that can be used to make predicitons (BDR)
        - Unlike ML, Bayes uses all information in the training set to make predictions.
        
        ---
        #### ML-BDR
        Lets consider the BDR under the "0-1" loss and an independent sample
        
        ML-BDR:
        - Pick i if
        $$
        i^*(x) = \text{argmax}_i P_{X|Y}(x|i;\theta_i^*)P_Y(i)\\
            \text{where } \theta_i^* = \text{arg max}_\theta P_{X|Y}(D|i, \theta)
        $$
        
        Two steps:
        - Find $\theta^*$
        - Plug into BDR
        
        All information is not captured by $\theta^*$ is lost, not used at decision time
        
        #### Bayes BDR
        Note that we know that information is lost
        - e.g. we cdan't even know how good of an estimate $\theta^*$ is unless we run multiple experiements and measure bias and vairance
        
        Under the Bayesian Framework, everything is conditioned on the training data
        - Denote $T = \{X_1, \dots, X_n\}$ the set of random variables from which the training sample is drawn
        
        B-BDR:
        - Pick $i$ if 
        $$
        i^*(x) = \text{argmax}_i P_{X|Y, T}(x|i, D_i)P_y(i)
        $$
        The decision is conditioned on the entire training set
        
        To compute the condition al probabilities, we use the marginalization equation
        $$
        P_{X|Y, T}(x|i, D_i) = \int P_{X|\Theta, Y, T}(x | \theta, i, D_i)P_{\Theta|Y, T}(\theta|i, D_i)d\theta
        $$
        
        Note 1: when the parameter value is known, x no longer depends on $T$, e.g. $X|\Theta ~ N(\theta, \sigma^2)$
        - We can simplify equation above into
        $$
        P_{X|Y, T}(x|i, D_i) = \int P_{X | \Theta, Y}(x | \theta, i) P_{\Theta| Y, T}(\theta|i, D_i)d\theta
        $$
        
        Note 2: once again can be done in two steps per class
        - Find $P_{\Theta|T}(\theta|D_i)$
        - Compute $P_{X|Y, T}(x|i, D_i)$ and plug into the BDR
        
        No training information is lost
        
        In summary:
        - Pick i if
        $$
        i^*(x) = \text{argmax}_i P_{X|Y, T}(x|i, D_i)P_Y(i)
        \text{where } P_{X|Y, T}(x|i, D_i) = \int P_{X|Y, \Theta}(x|i, \theta)P_{\Theta|Y, T}(\theta|i, D_i)d\theta
        $$
        Note:
        - as before the bottom equation is repeated for each class
        - Hence, we can drop the dependence on the class and consider the more general problem of estimating
        $$
        \boxed{
            P_{X|T}(x|D) = \int P_{X|\Theta}(x|\theta)P_{\Theta|T}(\theta|D) d\theta
        }
        $$
        
        #### The predictive distribution
        The distribution
        $$
        \boxed{
            P_{X|T}(x|D) = \int P_{X|\Theta}P_{\Theta|T}(\theta|D)d\theta
        }
        $$
        is known as the predictive distribution
        
        This follows from the fact that it allows use 
        - To predict the value of x
        - Given ALL the information available in the training set
        
        Note that it can also be written as
        $$
        P_{X|T}(x|D) = E_{\Theta|T}[P_{X|\Theta})(x|\theta)|T = D]
        $$
        - Since each parameter value defines a model
        - This is the expectation over all possible models
        - Each model is weighted by its posterior probability given the training data
        """)
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