Filter (signal processing)

a device or process that removes some unwanted components or features from a signal. Filtering is a class of signal processing, the defining feature of filters being the complete or partial suppression of some aspect of the signal. 

# Kalman & Bayesian Filters in Python Book Notes

Key points

- Multiple data points are more accurate than one data point, so throw nothing away no matter how inaccurate it is;
- always choose a number part way bewteen two data points to create a more accurate estimate;
- predict the next measurement and rate of change based on the current estimate and how much we think it will change
- the new estimate is then chosen as part way between the prediction and next measurement scaled by how much credence we give them.

Therefore, filtering is always a compromise between a(n) (informed) prediction and a (noisy) measurement, that will hopefully increase in accuracy over time, as we get more data.

Terminology & examples

System: an object we want to estimate. In some texts it's called a plant, which is combination of a process and an actuator, e.g. a scale.
Process: series of interrelated tasks that, together, transform inputs into a given output e.g. transforming gravity force on a scale to a number.
State: the current configuration or values of the system that we are interested in e.g., the weight on the scale.
Measurement: the measured value of the system e.g. the number on the scale.
State estimate: filter's estimate of the state e.g., of a scale.

In other words: the state should be unerstood as the actual value of the system. This value is usually hidden to us. If I stapped on a scale you'd then have a measurement. We cal this observable since you can directly observe this measurement. In contrast, you can never directly observe my weight, you can only measure it. Any estimation problem consists of forming an estimate of a hidden state via observable measurements. 

Process model: the mathematical model of the system, e.g., for a moving car the process model of the distance is: the distance at last measured timestep, plus velocity times time since that last timestep. This process model is not perfect as the velocity of a car can vary over a non-zero amount of time, the tires can slip on the road etc. 
Prediction: using a process model, we try to predict the current state, e.g. we try to predict the position of a car.
System error or Process error is the error in the model, e.g., the error caused by the process model's imperfections, i.e., the difference between the predicted position of the car and the actual position of the car.
System propagation: The prediction step. We use the process model to make a new state estimate. Because of the system error the estimate is imperfect. 
Epoch: one iteration of system propagation.

Say we want to predict the position of the train. Then, we have to take into account certain assumptions such as how much a train can slow down within a certain time frame and the system errors. These assumptions must be encoded in the credence we give prediction and the measurement. Then, the estimate is somewhere in between the prediction and measurement proportional to the credence we give them both. This can be a position in an n-dimensional space. The examples above, of the trian and scale are 1 dimensional.

Algorithm

Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
   Predict
3. Use system behaviour the predict the state at the next time step
4. Adjes belief to account for the uncertainty in prediction`
   Update
5. Get a measurement and associated belief about its accuracy
6. Compute residual between estimated state and measurement
7. New estimate is somewhere on the residual (line between measurement and estimate)

g-h filter

estimate = prediction + g*residual
new gain = old gain + h * 1/dt * residual

where 

- residual  = measurement - prediction;
- prediction = last estimate + gain * dt;
- g describes how sure we are about the measurement, i.e. how much we want to fit the measurement error (residual), or how accurate we predict the measurement to be. The more sure we are about the measurement (residual \approx 0), the higher g should be.
- hdecribes the prediction of how fast dxchanges over time.

Kalman filter

We learned that the sum of 2 gaussians is:

and the product of 2 gaussians is

and we know that the posterior is:

As long as everything is gaussian we can use measurements and predictions to update the Kalman filter:

```
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
```

```
print('PREDICT\t\t\tUPDATE')
print('     x      var\t\t  z\t    x      var')

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

KF algorithm

```
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

$R$ is measurement noise
$Q$ is process noise
$P$ is state variance
$z$ is measurement.
Kalman gain $K = P/(P+R)$

Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state
   Predict
3. use system behavior to predict state at the net time step
4. Adjust belief to account for the uncertainty in prediction
   Update
5. Get a measruement and associated belief about its accuracy
6. COmpute residual between estimated state and measurement
7. COmpute scaling factor based on whether the measurement or prediction is more accuracte
8. set state between the prediction and measurement based on scaling factor
9. update belief in the state based on how certain we are in the measurement.

**Equations**

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

**Update**

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





