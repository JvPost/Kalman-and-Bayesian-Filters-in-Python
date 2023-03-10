# Prediction equations

Simply as described above.

The algorithm stays the same, we just add more dimensions now.

<u><strong>Predict</strong></u>

$$
\begin{array}{|l|l|l|}\hline\text{Univariate} & \text{Univariate} & \text{Multivariate}\\& \text{(Kalman form)} & \\\hline\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q \\\hline\end{array}
$$

$\mathbf x$, $\mathbf P$ are the state mean and covariance. They correspond to $x$ and $\sigma^2$.

$\mathbf F$ is the *state transition function*. When multiplied by $\mathbf x$ it computes the prior.

$\mathbf Q$ is the process covariance. It corresponds to $\sigma^2_{f_x}$.

$\mathbf B$ and $\mathbf u$ let us model control inputs to the system.

<u><strong>Update</strong></u>

$$
\begin{array}{|l|l|l|}\hline\text{Univariate} & \text{Univariate} & \text{Multivariate}\\& \text{(Kalman form)} & \\\hline& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\& K = \frac{\bar P}{\bar P+R}&\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P &\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\\hline\end{array}
$$

$\mathbf H$ is the measurement function. We haven't seen this yet in this book and I'll explain it later. If you mentally remove $\mathbf H $ from the equations, you should be able to see these equations are similar as well.

$\mathbf z,\, \mathbf R$ are the measurement mean and noise covariance. They correspond to $z$ and $\sigma_z^2$ in the univariate filter (I've substituted $\mu$ with $x$ for the univariate equations to make the notation as similar as possible).

$\mathbf y$ and $\mathbf K$ are the residual and Kalman gain.

Our job is to design a state ($\mathbf{x}, \mathbf{P}$), the process ($\mathbf{F}, \mathbf{Q}$), the measurement ($\mathbf{z}, \mathbf{R}$), and the measurement function $\mathbf{H}$. If the system has control inputs, such as a robot, you will also design $\mathbf{B}$ and $\mathbf{u}$.
