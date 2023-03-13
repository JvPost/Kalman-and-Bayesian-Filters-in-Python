# Filter (signal processing)

A device or process that removes some unwanted components or features from a signal. Filtering is a class of signal processing, the defining feature of filters being the complete or partial suppression of some aspect of the signal. From [Filter (signal processing) - Wikipedia](https://en.wikipedia.org/wiki/Filter_(signal_processing))

# Kalman & Bayesian Filters in Python Book Notes

Key points

- Multiple data points are more accurate than one data point, so throw nothing away no matter how inaccurate it is;
- always choose a number part way between two data points to create a more accurate estimate;
- predict the next measurement and rate of change based on the current estimate and how much we think it will change;
- the new estimate is then chosen as part way between the prediction and next measurement scaled by how much credence we give them.

Therefore, filtering is always a compromise between a(n) (informed) prediction and a (noisy) measurement, that will hopefully increase in accuracy over time, as we get more data.

## Terminology & examples

**System**: an object we want to estimate. In some texts it's called a [Plant (control theory) - Wikipedia](https://en.wikipedia.org/wiki/Plant_(control_theory)), which is combination of a process and an actuator, e.g. a scale. 
**Process**: series of interrelated tasks that, together, transform inputs into a given output e.g. transforming gravity force on a scale to a number.
**State**: the current configuration or values of the system that we are interested in e.g., the weight on the scale.
**Measurement**: the measured value of the system e.g. the number on the scale.
**State estimate**: filter's estimate of the state e.g., of a scale.

**In other words**: the state should be understood as the actual value of the system. This value is usually hidden to us. If I step on a scale you'd have a measurement. We call this observable since you can directly observe this measurement. In contrast, you can never directly observe my weight, you can only measure it. Any estimation problem consists of forming an estimate of a hidden state via observable measurements. 

**Process model**: the mathematical model of the system, e.g., for a moving car the process model of the distance is: the distance at last measured timestep, plus velocity times time since that last timestep. This process model is not perfect as the velocity of a car can vary over a non-zero amount of time, the tires can slip on the road etc. 
**Prediction**: using a process model, we try to predict the current state, e.g. we try to predict the position of a car.
**System error** or **Process error** is the error in the model, e.g., the error caused by the process model's imperfections, i.e., the difference between the predicted position of the car and the actual position of the car.
**System propagation**: The prediction step. We use the process model to make a new state estimate. Because of the system error the estimate is imperfect. 
**Epoch**: one iteration of system propagation.

Say we want to predict the position of the train. Then, we have to take into account certain assumptions such as how much a train can slow down within a certain time frame and the system errors. These assumptions must be encoded in the credence we give prediction and the measurement. Then, the estimate is somewhere in between the prediction and measurement proportional to the credence we give them both. This can be a position in an n-dimensional space. The examples above, of the trian and scale are 1 dimensional.

## Algorithm

**Initialization**

1. Initialize the state of the filter
2. Initialize our belief in the state

**Predict**

1. Use system behaviour the predict the state at the next time step
2. Adjust belief to account for the uncertainty in prediction

**Update**

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. New estimate is somewhere on the residual (line between measurement and estimate)

# g-h filter

$\text{estimate} = \text{prediction} + g \cdot \text{residual}$
$\text{new gain} = \text{old gain} + h \frac{1}{dt} \cdot \text{residual}$

where 

- $\text{residual}  = \text{measurement} - \text{prediction}$
- $\text{prediction} = \text{last estimate} + \text{gain} \cdot dt$
- $g$ describes how sure we are about the measurement, i.e. how much we want to fit the measurement error (residual), or how accurate we predict the measurement to be. The more sure we are about the measurement ($\text{residual} \approx 0$), the higher $g $should be.
- $h$ decribes the prediction of how fast the system exchanges over time.

# 1D Kalman filter

We know that the <u>sum of two gaussians</u> is

$$
\mathcal{N}(\mu_1, \sigma_1^2) + \mathcal{N}(\mu_2, \sigma_2^2) = \\
\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$

We use this for predictions, as prediction are simply linear operations

and the <u>product of 2 gaussians</u> is

$$
\mathcal{N}(\mu_1, \sigma_1^2) \cdot \mathcal{N}(\mu_2, \sigma_2^2) = \\
\mathcal{N}\left( \frac{\sigma_1^2\mu_2 + \sigma_2^2\mu_1}{\sigma_1^2 + \sigma_2^2},
\frac{\sigma_1^2\sigma_2^2}{\sigma_1^2 + \sigma_2^2}
  \right)
$$

and we know that the posterior is:

$\text{posterior} = \frac{\text{likelihood} \cdot \text{prior}}{\text{marginal}}$

Therefore the update step is simply 

$$
\mathcal{N}(\bar\mu, \bar\sigma^2) \cdot \mathcal{N}(\mu_z, \sigma_z^2) = \\
\mathcal{N}\left( \frac{\bar\sigma^2\mu_z + \sigma_z^2\bar\mu_1}{\bar\sigma^2 + \sigma_z^2},
\frac{\bar\sigma^2\sigma_z^2}{\bar\sigma^2 + \sigma_z^2}
  \right)
$$

where $\bar\mu$ and $\bar\sigma^2$ denote prior parameters and $\mu_z$ and $\sigma^2_z$ denote measurement model parameters. Remember, both the prior and measurement have noise. Then in the subsequent steps the posterior becomes the new prior. 

**Likelihood**

So, we get mean and covariance vectors from multiplying the prior and the likelihood.

Remember, the prior is the posterior we previously calculated, and the **likelihood is the likely position given the measurement z**. In these notes I denote priors with a bar and likelihoods with the subscript $z$.

This description of the system corresponds with the definition of the likelihood function. The likelihood function is the probability of the data given the model parameters. In this system the model parameters are the measurements $z$ we take, because they are a description of the reality. The data is the actual state of the system $x$.

Example: given that the die rolled 6 three times, what is the likelihood that the die is fair? Or in our case: how likely are the measures in the current state $p(z | x)$. And we want to find 

$$
p(x|z) = \frac{p(z|x)p(x)}{p(z)}
$$



As long as everything is gaussian we can use measurements and predictions to update the Kalman filter:

```python
import kf_book.kf_internal as kf_internal
from kf_book.kf_internal import DogSimulation

np.random.seed(13)

process_var = 1. # variance in the dog's movement
sensor_var = 2. # variance in the sensor

x = gaussian(0., 20.**2)  # dog's position, N(0, 20**2)
velocity = 1
dt = 1. # time step in seconds
process_model = gaussian(velocity*dt, process_var) # displacement to add to x

# simulate dog and get measurements
dog = DogSimulation(
    x0=x.mean, 
    velocity=process_model.mean, 
    measurement_var=sensor_var, 
    process_var=process_model.var)

# create list of measurements
zs = [dog.move_and_sense() for _ in range(10)]

# perform Kalman filter on measurement z
for z in zs:    
    prior = predict(x, process_model)
    likelihood = gaussian(z, sensor_var)
    x = update(prior, likelihood)

    kf_internal.print_gh(prior, x, z)

print()
print(f'final estimate:        {x.mean:10.3f}')
print(f'actual final position: {dog.x:10.3f}')
```

For each iteration of the loop we forma prior, take a measurement, form a likelihood from the measurement, and then incorporate the likelihood into the prior.

## Kalman gain derivation

$$
\begin{aligned}
\mu^* &= \frac{\bar \sigma^2\mu_z+\sigma_z^2\bar\mu}{\bar\sigma^2 + \sigma_z^2} \\
&= \frac{1}{\bar\sigma^2 + \sigma_z^2} \bar\sigma^2\mu_z+ 
\frac{1}{\bar\sigma^2 + \sigma_z^2} \sigma^2_z\bar\mu \\
&= \frac{\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2} \mu_z+ 
\frac{\sigma^2_z}{\bar\sigma^2 + \sigma_z^2} \bar\mu \\
&= \frac{\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2} \mu_z + 
\left(1- \frac{\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2} \right) \bar\mu \\
&= K\mu_z + (1-K)\bar\mu
\end{aligned}
$$

where $\mu^*$ is the posterior mean and $K=\frac{\bar\sigma^2}{\bar\sigma^2 + \sigma_z^2}$ is called **Kalman gain**

$$
\begin{aligned}
 & K \mu_z + (1-K)\bar\mu \\
 &= K \mu_z + \bar\mu -K\bar\mu \\
&= \bar\mu + K(\mu_z - \bar\mu)
\end{aligned}
$$

The posterior variance ${\sigma^2}^*$ can also be defined in terms of $K$:

$$
\begin{aligned}
{\sigma^2}^* &= \frac{\bar\sigma^2 \sigma_z^2}{\bar \sigma^2 + \sigma^2_z} \\
&= \sigma^2_z \frac{\bar\sigma^2 }{\bar \sigma^2 + \sigma^2_z} \\
&= \sigma^2_z K \\
&= \bar\sigma^2 (1-K)
\end{aligned}
$$

## KF algorithm

```python
def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)

def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)
```

$$
$$

$R$ is measurement noise: $\sigma^2_z$

$Q$ is process noise: $\bar\sigma^2 + \sigma_z^2$

P is state variance:$ \bar\sigma^2$
$z$ is measurement: $z \sim \mathcal{N}(\mu_z, \sigma_z^2)$

Kalman gain $K = P/(P+R)$

**Initialization**

1. Initialize the state of the filter
2. Initialize our belief in the state

**Predict**

1. use system behavior to predict state at the net time step
2. Adjust belief to account for the uncertainty in prediction

**Update**

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement or prediction is more accuracte
4. Set state between the prediction and measurement based on scaling factor
5. Update belief in the state based on how certain we are in the measurement.

## Equations

**<u>Predict</u>**

$$

\begin{array}{|l|l|l|}
\hline
\text{Equation} & \text{Implementation} & \text{Kalman Form}\\
\hline
 \bar x = x + f_x & \bar\mu = \mu + \mu_{f_x} & \bar x = x + dx\\
& \bar\sigma^2 = \sigma^2 + \sigma_{f_x}^2 & \bar P = P + Q\\
\hline
\end{array}
$$

**<u>Update</u>**

$$
\begin{array}{|l|l|l|}
\hline
\text{Equation} & \text{Implementation}& \text{Kalman Form}\\
\hline
 x = \| \mathcal L\bar x\| & y = z - \bar\mu & y = z - \bar x\\
 & K = \frac {\bar\sigma^2} {\bar\sigma^2 + \sigma_z^2} & K = \frac {\bar P}{\bar P+R}\\
 & \mu = \bar \mu + Ky & x = \bar x + Ky\\
 & \sigma^2 = \frac {\bar\sigma^2 \sigma_z^2} {\bar\sigma^2 + \sigma_z^2} & P = (1-K)\bar P\\
\hline
\end{array}
$$

# nD Kalman filter

The algorithm stays the same, we just add more dimensions now.

<u>**Predict**</u>

$$
\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q \\
\hline
\end{array}
$$

$\mathbf x,\, \mathbf P$ are the state mean and covariance. They correspond to $x$ and $\sigma^2$.

$\mathbf F$ is the <u>state transition function</u>. When multiplied by $\bf x$ it computes the prior.

$\mathbf Q$ is the process covariance. It corresponds to $\sigma^2_{f_x}$.

$\mathbf B$ and $\mathbf u$ are new to us. They let us model control inputs to the system.

<u>**Update**</u>

$$
\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\
& K = \frac{\bar P}{\bar P+R}&
\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\
\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\
\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P &
\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\
\hline
\end{array}
$$

$\mathbf H$ is the measurement function.

$\mathbf z,\, \mathbf R$ are the measurement mean and noise covariance. They correspond to $z$ and $\sigma_z^2$ in the univariate filter.

$\mathbf y$ and $\mathbf K$ are the residual and Kalman gain.

Our job is to design a state $(\mathbf{x}, \mathbf{P})$, the process $(\mathbf{F}, \mathbf{Q})$, the measurement $(\mathbf{z}, \mathbf{R})$, and the measurement function $\mathbf{H}$. If the system has control inputs, such as a robot, you will also design $\mathbf{B}$ and $\mathbf{u}$.

**Observed variables**: directly measured by sensors, e.g., position

**Hidden variable**: inferred from the from the <u>observed variables</u>, e.g., velocity and accelaration

## Prediction

### Design state covariance

**State covariance** $\mathbf{P}$ is simply the variance of every observed and/or hidden variables, over de diagional of a squared matrix. The non diagonal parts are the covariances.

### Design process model

**Example** system of moving object with constant velocity, then we try to find the <u>state transition matrix \ state transition function </u> $\mathbf{F}$.

$$
\bar x = x + \dot x\Delta t \\
\bar{\dot x}  = \dot x \\
\mathbf{x} = (x, \dot x)^\text{T}
$$

then the process model can be found by creating a system of linear equations:

$$
\begin{cases}
\begin{aligned}
\bar x &= x + \dot x \Delta t \\
\bar{\dot x} &= \dot x
\end{aligned}
\end{cases}
$$

$$
\begin{cases}
\begin{aligned}
\bar x &= 1x + &\Delta t\, \dot x \\
\bar{\dot x} &=0x + &1\, \dot x
\end{aligned}
\end{cases}
$$

$$
\begin{aligned}
\begin{bmatrix}\bar x \\ \bar{\dot x}\end{bmatrix} &= \begin{bmatrix}1&\Delta t  \\ 0&1\end{bmatrix}  \begin{bmatrix}x \\ \dot x\end{bmatrix}\\
\mathbf{\bar x} &= \mathbf{Fx}
\end{aligned}
$$

### Predict

We have now found the state covariance $\mathbf{P}$ and state transition matrix $\mathbf{F}$. Assuming no process noise and control functions by setting $\mathbf{Q}=0$ and $\mathbf{u}=0$, respectively. We can now make a predict step using the equations from above:

$$
\mathbf{\bar x = Fx} \\
\mathbf{\bar P = FPF}^\mathsf{T}

$$

### Design Process Noise

Random forces can influence a system, we can model this for example by adding white noise:

$$
\mathbf{\dot x = }f(\mathbf{x}) + w
$$

where $w \sim \mathcal N(0, \sigma_w^2)$

### Design the Control function

$$
\Delta\mathbf x = \mathbf{Bu}
$$

Here $\mathbf{u}$ is the <u>control input</u> and $\mathbf{B}$ is the <u>control model</u>.

### Summary

Therefore, to make a prediction we have to specify

$\mathbf{x, P}$: the state mean and covariance

$\mathbf{F, Q}$: the process model and noise covariance

$\mathbf{B, u}$: optionally, the control input and function

## Update Step

### Design the measurement function

The measurement function $\mathbf{H}$ transforms state into measurement, i.e. temperature into volts. Then, we calculate the residual by:

$$
\mathbf y = \mathbf z - \mathbf{H \bar x}
$$

where $\mathbf y$ is the residual, $\mathbf{\bar x}$ is the prior, $\mathbf z$ is the measurement, and $\mathbf H$ is the measurement function. 

### Design the measurement

$$
\mathbf z = \begin{bmatrix}z_1 \\ z_2\end{bmatrix}
$$

$$
\mathbf R = \begin{bmatrix} \sigma_{z_1}^2 & \text{cov}(\sigma_{z_1}^2, \sigma_{z_2}^2) \\ \text{cov}(\sigma_{z_1}^2, \sigma_{z_2}^2) & \sigma_{z_2}^2 \end{bmatrix}
$$

## Kalman filter Equations

### Prediction equations

Simply as described above.

The algorithm stays the same, we just add more dimensions now.

<u><strong>Predict</strong></u>

$$
\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
\bar \mu = \mu + \mu_{f_x} & \bar x = x + dx & \bar{\mathbf x} = \mathbf{Fx} + \mathbf{Bu}\\
\bar\sigma^2 = \sigma_x^2 + \sigma_{f_x}^2 & \bar P = P + Q & \bar{\mathbf P} = \mathbf{FPF}^\mathsf T + \mathbf Q \\
\hline
\end{array}
$$

$\mathbf x$,  $\mathbf P$ are the state mean and covariance. They correspond to $x$ and $\sigma^2$.

$\mathbf F$ is the *state transition function*. When multiplied by $\mathbf x$ it computes the prior.

$\mathbf Q$ is the process covariance. It corresponds to $\sigma^2_{f_x}$.

$\mathbf B$ and $\mathbf u$ let us model control inputs to the system.

<u><strong>Update</strong></u>

$$
\begin{array}{|l|l|l|}
\hline
\text{Univariate} & \text{Univariate} & \text{Multivariate}\\
& \text{(Kalman form)} & \\
\hline
& y = z - \bar x & \mathbf y = \mathbf z - \mathbf{H\bar x} \\
& K = \frac{\bar P}{\bar P+R}&
\mathbf K = \mathbf{\bar{P}H}^\mathsf T (\mathbf{H\bar{P}H}^\mathsf T + \mathbf R)^{-1} \\
\mu=\frac{\bar\sigma^2\, \mu_z + \sigma_z^2 \, \bar\mu} {\bar\sigma^2 + \sigma_z^2} & x = \bar x + Ky & \mathbf x = \bar{\mathbf x} + \mathbf{Ky} \\
\sigma^2 = \frac{\sigma_1^2\sigma_2^2}{\sigma_1^2+\sigma_2^2} & P = (1-K)\bar P &
\mathbf P = (\mathbf I - \mathbf{KH})\mathbf{\bar{P}} \\
\hline
\end{array}
$$

$\mathbf H$ is the measurement function. We haven't seen this yet in this book and I'll explain it later. If you mentally remove $\mathbf H $ from the equations, you should be able to see these equations are similar as well.

$\mathbf z,\, \mathbf R$ are the measurement mean and noise covariance. They correspond to $z$ and $\sigma_z^2$ in the univariate filter (I've substituted $\mu$ with $x$ for the univariate equations to make the notation as similar as possible).

$\mathbf y$ and $\mathbf K$ are the residual and Kalman gain.

Our job is to design a state ($\mathbf{x}, \mathbf{P}$), the process ($\mathbf{F}, \mathbf{Q}$), the measurement ($\mathbf{z}, \mathbf{R}$), and the measurement function $\mathbf{H}$. If the system has control inputs, such as a robot, you will also design $\mathbf{B}$ and $\mathbf{u}$.

### Update equations

**<u>System uncertainty</u>**

$\textbf{S} = \mathbf{H\bar PH}^\mathsf T + \mathbf R$

We have to work in the measurement space, therefore the *prior* must be projected into it. So $\bf S$ is the prior state, transformed into the measurement space, plus measurement noise covariance.

<u>**Kalman Gain**</u>

Remember: 

$$
\mu =\frac{\bar\sigma^2 \mu_z + \sigma_\mathtt{z}^2 \bar\mu} {\bar\sigma^2 + \sigma_\mathtt{z}^2}\\
\mu = (1-K)\bar\mu + K\mu_\mathtt{z}\\

K = \frac {\bar\sigma^2} {\bar\sigma^2 + \sigma_z^2}
$$

$K$ is the <u>Kalman gain</u>, a ratio of how much credence we give the prediction and measurement. 

Translating it from univariate to multi variate $\bf S$ corresponds to $\bar \sigma^2 + \sigma_z^2$, therefore 

$$
\bf K = \bf{\bar P} H^\mathsf T \bf S^{-1}
$$

Since  $\textbf{S} = \mathbf{H\bar PH}^\mathsf T + \mathbf R$ we can also write it as:

$$
\bf K = \bf{\bar P} H^\mathsf T (\bf{H \bar P H^\mathsf T + R})^{-1}
$$

**<u>Residual</u>**

Simply the measurement $\bf z$ minus the measurement space projected state $\bf{H\bar x}$ giving

$$
\bf{y = z - H\bar x}
$$

<u>**State update**</u>

$$
\bf{x = \bar x + Ky}
$$

More clearly shown by

$$
\begin{aligned}
\mathbf x &= \mathbf{\bar x} + \mathbf{Ky} \\
&= \mathbf{\bar x} +\mathbf K(\mathbf z - \mathbf{H\bar x}) \\
&= (\mathbf I - \mathbf{KH})\mathbf{\bar x} + \mathbf{Kz}
\end{aligned}
$$

Here you also see that we've chosen to write down $\bf \bar x$ and $\bf z$, rather than $\bf \bar \mu$ and $\mathbf \mu_z$. But of course though, when write down $\bf \bar x$ we're talking about the mean. This will be clear from the code.

**<u>Covariance update</u>**

$$
\bf{P = (I-KH)\bar P}
$$

### Summary

$$
\begin{aligned}

\text{Predict Step}\\

\mathbf{\bar x} &= \mathbf{F x} + \mathbf{B u} \\

\mathbf{\bar P} &= \mathbf{FP{F}}^\mathsf T + \mathbf Q \\

\\

\text{Update Step}\\

\textbf{S} &= \mathbf{H\bar PH}^\mathsf T + \mathbf R \\

\mathbf K &= \mathbf{\bar PH}^\mathsf T \mathbf{S}^{-1} \\

\textbf{y} &= \mathbf z - \mathbf{H \bar x} \\

\mathbf x &=\mathbf{\bar x} +\mathbf{K\textbf{y}} \\

\mathbf P &= (\mathbf{I}-\mathbf{KH})\mathbf{\bar P}

\end{aligned}

$$

### Paper notation

$$
\begin{aligned}

\hat{\mathbf x}_{k\mid k-1} &= \mathbf F_k\hat{\mathbf x}_{k-1\mid k-1} + \mathbf B_k \mathbf u_k  \\

\mathbf P_{k\mid k-1} &=  \mathbf F_k \mathbf P_{k-1\mid k-1} \mathbf F_k^\mathsf T + \mathbf Q_k \\            

\tilde{\mathbf y}_k &= \mathbf z_k - \mathbf H_k\hat{\mathbf x}_{k\mid k-1}\\

\mathbf{S}_k &= \mathbf H_k \mathbf P_{k\mid k-1} \mathbf H_k^\mathsf T + \mathbf R_k \\

\mathbf K_k &= \mathbf P_{k\mid k-1}\mathbf H_k^\mathsf T \mathbf{S}_k^{-1}\\

\hat{\mathbf x}_{k\mid k} &= \hat{\mathbf x}_{k\mid k-1} + \mathbf K_k\tilde{\mathbf y}_k\\

\mathbf P_{k|k} &= (I - \mathbf K_k \mathbf H_k) \mathbf P_{k|k-1}

\\\end{aligned}

$$

## Filter initialization

# Designing Kalman Filters

Order of design:

1. State $\bf x$ (one dimension for every (hidden) variable)

2. State transition matrix $\bf F$

3. Process Noise Matrix $\bf Q$

4. Control function $\bf B$

5. Measurement function $\bf H$

6. Measurement noise $\bf R$

7. Initial conditions $\mathbf{x}_0$ and $\mathbf{P}_0$

## n-order kalman filters

You can decide incorporate different orders of derivation or polynomial into your state model e.g. $\mathbf{x} = [x, \dot x, \ddot x]$, where $x$ is position, $\dot x$ is velocity and $\ddot x$ is accelaration. Let's say we have we're tracking a filtering an object that has a constant speed. You might conjecture that incorporating $\ddot x$ would be good, because more information is always better and if $\ddot x$ is 0 anyways, it doesn't matter. However, this is not true in this case, because the accelaration sensor also has noise, which can be wrongly interpreted by the filter. Additionally, hsigh noise in position and velocity may be interpreted as accelaration. But we said $\dot x$ was constant, and therefore $\ddot x = 0$. Therefore, setting the filter order as high as possible it not always the best choice. Sometimes we need to throw away information.

For example, consider a system of 2 points. For 0th or 1st polyonomials the optimal approximation is easy: it's the straight light that minimizes the error. However, increase the order and ther are an infinite amount of answers that minimize the error. Therefore, you need a filter whose order matches the system's order. 

With that said, a lower oder filter can track a higher order process so long as you add enough process noise and you keep the descretization period small (100 samples a second are useually locally linear).

# Unscented Kalman Filters

**Sigma points**: data points sampled from a normal transformed by some non-linear function s.t.:

$$
\{\text{x}\} \sim \mathcal N(\mu, \sigma^2)\\
y_i = f(x_i)
$$

where $f()$ is a non-linear function and $y_i$ is a sigma point.
