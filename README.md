# WAR!

This is my son Oscar.  He is five.  

<img style="max-width: 500px; height: auto; " src="oscar.jpg"/>

He recently learned how to play [war](https://en.wikipedia.org/wiki/War_(card_game)).  

The game is *not quite* deterministic, and actually *not quite* a game.  It's not really a game because it involves no choice on the part of the players, and it's not really deterministic, because the ordering of the winners cards, added to the bottom of their pile, is not determined by any rule.  

Naturally I became curious what the game's turn count distribution would look like.  You gotta figure that it's possible -- though very unlikely -- that it could resolve in a single turn:  one huge chain war.  It also seems plausible that it could go on infinitely; there would have to exist some combination of the initial ordering and orderings of all the win pots such that no player's pile ever goes all the way down to zero.  In practical terms, there's probably a very long right tail.

So let's see.  I'm going to import the class `War` from [war.py](https://github.com/cranedroesch/war/blob/main/war.py) -- which I wrote on an Amtrak train while Oscar played war with [his mother](https://github.com/ertzeid) (who turned me on to python `queue`s and wrote some of the `War` class while I took a turn playing with Oscar) -- and use it to simulate lots of games.  I'll then count their turn distribution, and see whether it corresponds with any known, stardard count distributions, like poisson or negative binomial or something.  I doubt it, but let's see.


```python
from war import War, war_sim
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import numpy as np
from scipy.stats import poisson
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

Here's an example with verbose mode turned on.  I cherrypicked this seed to not have too big of a turn count.


```python
game = War(verbose = True, seed = 26)
game.play()
```

    Starting turn 1
    10, 9, p1 wins [10, 9]
    Adding pot [10, 9] to pile <queue.Queue object at 0x7fdbc0fb6490> of size 25
    Turn end status: turn 1 p1: 27, p2: 25
    Starting turn 2
    
    --SNIP--
    
    Starting turn 29
    WAR!!! 13, 13, pot [13, 13]
    WAR! Pot: [13, 13]
    Added 10 to pot; pot now [13, 13, 10]
    Added 6 to pot; pot now [13, 13, 10, 6]
    Added 5 to pot; pot now [13, 13, 10, 6, 5]
    Added 9 to pot; pot now [13, 13, 10, 6, 5, 9]
    Added 11 to pot; pot now [13, 13, 10, 6, 5, 9, 11]
    Added 8 to pot; pot now [13, 13, 10, 6, 5, 9, 11, 8]
    Added 13 to pot; pot now [13, 13, 10, 6, 5, 9, 11, 8, 13]
    Added 9 to pot; pot now [13, 13, 10, 6, 5, 9, 11, 8, 13, 9]
    13, 9: p1 wins pot [13, 13, 10, 6, 5, 9, 11, 8, 13, 9]
    Adding pot [13, 13, 10, 6, 5, 9, 11, 8, 13, 9] to pile <queue.Queue object at 0x7fdbc0fb6490> of size 29
    turn 29 p1: 39, p2: 13
    Turn end status: turn 29 p1: 39, p2: 13
    
    --SNIP--

    Starting turn 52
    13, 4, p1 wins [13, 4]
    Adding pot [13, 4] to pile <queue.Queue object at 0x7fdbc0fb6490> of size 50
    Turn end status: turn 52 p1: 52, p2: 0
    GAME OVER: turn 52 p1: 52, p2: 0

    52



Now we'll simulate lots of games.  


```python
pool = mp.Pool(mp.cpu_count())
counts = pool.map(war_sim, range(50000))
pool.close()
```


```python
# I'm going to reuse this a lot...
def mainplot():
    fig, ax = plt.subplots(ncols = 2, figsize = (16, 6))
    fig.patch.set_facecolor('lightblue')
    ax[0].hist(counts, edgecolor = 'black', density = True, bins = 100)
    ax[0].set_xlabel('Turn counts')
    ax[0].set_title('Distribution of war turn counts')
    sns.kdeplot(np.log10(counts), ax=ax[1], label = 'kernel density')
    ax[1].set_xlabel('log10(Turn counts)')
    ax[1].set_title('Distribution of war turn counts (orders of magnitude)')
    ax[1].legend()
    return fig, ax
fig, ax = mainplot()
```


    
![png](README_files/README_6_0.png)
    



```python
mu = round(np.mean(counts))
q = np.quantile(counts, [.05, .25,.5, .75, .95])
print(f"The average is {mu}.  \n5% of the time, games are {q[0]} turns or fewer."\
      f"\n5% of the time, games are {q[4]} turns or more"\
      f"\nA quarter of the time, the games are {q[1]} turns or fewer."
      f"\nAnother quarter of the time, the games are {q[3]} turns or fewer."
      f"\nThe median game is {q[2]} turns.")
```

    The average is 271.  
    5% of the time, games are 52.0 turns or fewer.
    5% of the time, games are 711.0 turns or more
    A quarter of the time, the games are 114.0 turns or fewer.
    Another quarter of the time, the games are 356.0 turns or fewer.
    The median game is 204.5 turns.


## It is poisson?  
Doesn't really look like it.  But it's easy to check; poisson's single parameter is the mean of the counts.  


```python
pmf = np.arange(poisson.ppf(0.01, np.mean(counts)),
                poisson.ppf(0.99, np.mean(counts)))
grid = list(range(min(counts), max(counts)))
```


```python
def log10pmf(x: np.ndarray):
    '''
    take a pmf and transform it to be a valid pmf on the log10 scale
    '''
    y = 10**x
    y-=y.min()
    y/=y.sum()
    return y
```


```python
log10pmf(poisson.pmf(grid, np.mean(counts))).sum()
```




    1.0000000000000002




```python
fig, ax = mainplot()
ax[0].plot(grid, poisson.pmf(grid, np.mean(counts)), label = 'mle poisson')
ax[0].legend()
cx1 = ax[1].twinx()
logpmf =log10pmf(poisson.pmf(grid, np.mean(counts)))
cx1.plot(np.log10(grid), logpmf, color = 'orange')
cx1.set_ylim(0, max(logpmf))

```




    (0.0, 0.024458575221015896)




    
![png](README_files/README_12_1.png)
    


Nope

### Is it negative binomial?

It certainly isn't generated by the standard definition of the number of successes before $r$ failures, with the probability of each trial being $p$.  But let's give it a whirl and see if it's a good approximation.  I'm going to use tensorflow rather than scipy because (1) I've got tensorflow code lying around, and (2) why not?


```python
r_init = 1.
p_init = .9
rate = tf.Variable([r_init])
prob = tf.Variable([p_init])
dist = tfd.NegativeBinomial(total_count = rate, probs = prob)
optimizer = tf.optimizers.Adam()
```


```python
def loss(dist, data):
    total_log_prob = -tf.reduce_mean(dist.log_prob(data))
    return total_log_prob
     
def train_step(dist, data):
    with tf.GradientTape() as g:
        loss_value = loss(dist, data)
        grads = g.gradient(loss_value, dist.trainable_variables)
    optimizer.apply_gradients(zip(grads, dist.trainable_variables))
    return loss_value
```


```python
lossvec = []
for i in range(1000):
    lossvec.append(train_step(dist, counts))
```


```python
plt.plot(np.log10(lossvec))
```




    [<matplotlib.lines.Line2D at 0x7fdbd05a46a0>]




    
![png](README_files/README_18_1.png)
    


Note that the loss should be near zero if these were in fact random draws from some negbin distribution; we've got a large sample, and there's no noise.

The estimates:


```python

dist.trainable_variables
```




    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([0.9954184], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.2453775], dtype=float32)>)




```python
fig, ax = mainplot()
ax[0].plot(grid, dist.prob(grid), label = 'MLE negbin')
ax[0].legend()
cx1 = ax[1].twinx()
logpmf = log10pmf(dist.prob(grid).numpy())
cx1.plot(np.log10(grid), logpmf, color = 'orange')
cx1.set_ylim(0, max(logpmf))
```




    (0.0, 0.0028377037961035967)




    
![png](README_files/README_21_1.png)
    


Not bad, but not great.  It's not a negbin process.

## Maybe Gamma?  
It's a count variable, rather than a positive one.  But 


```python
a_init = 10.
b_init = .1
alpha = tf.Variable([a_init])
beta = tf.Variable([b_init])
dist = tfd.Gamma(concentration = alpha, rate = beta)
optimizer = tf.optimizers.Adam()
```


```python
def loss(dist, data):
    total_log_prob = -tf.reduce_mean(dist.log_prob(data))
    return total_log_prob
     
def train_step(dist, data):
    with tf.GradientTape() as g:
        loss_value = loss(dist, data)
        grads = g.gradient(loss_value, dist.trainable_variables)
    optimizer.apply_gradients(zip(grads, dist.trainable_variables))
    return loss_value
```


```python
lossvec = []
for i in range(5000):
    lossvec.append(train_step(dist, counts))
```


```python
plt.plot(np.log10(lossvec))
```




    [<matplotlib.lines.Line2D at 0x7fdba08d81f0>]




    
![png](README_files/README_27_1.png)
    


Not done...


```python
for i in range(5000):
    lossvec.append(train_step(dist, counts))
```


```python
plt.plot(np.log10(lossvec))
```




    [<matplotlib.lines.Line2D at 0x7fdbb0b734c0>]




    
![png](README_files/README_30_1.png)
    


A little more..


```python
for i in range(1000):
    lossvec.append(train_step(dist, counts))
```


```python
plt.plot(np.log10(lossvec))
```




    [<matplotlib.lines.Line2D at 0x7fdba0a2d2b0>]




    
![png](README_files/README_33_1.png)
    



```python
dist.trainable_variables
```




    (<tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([1.8090765], dtype=float32)>,
     <tf.Variable 'Variable:0' shape=(1,) dtype=float32, numpy=array([0.00668677], dtype=float32)>)




```python
fig, ax = mainplot()
ax[0].plot(grid, dist.prob(grid), label = 'MLE gamma')
ax[0].legend()
cx1 = ax[1].twinx()
cx1.plot(np.log10(grid), 10**(dist.prob(grid)), color = 'orange')
```




    [<matplotlib.lines.Line2D at 0x7fdb780654f0>]




    
![png](README_files/README_35_1.png)
    


### What have we learned?

We've learned that the turn counts in the game of war follows some distribution that isn't poisson, negative binomial, or gamma.  It's slightly left-skewed in log space, so it's not lognormal, though that probably wouldn't be a bad approximation.

What would the analytical distribution look like?  A mess, clearly.  It'd have to be the summation of the product of a bunch of probabilities player 1's card is greater than player 2's card, multiplied by the probability that either player has one card left, which itself would be conditional on all preceeding terms.  And that's not thinking through what happens when there is a war.  You'd want to start with the simplest case:

$$
W(d = 2, n = 1)
$$

where $d$ is the deck size and the number of suits $n = 1$ guarantees that there will never be any wars.  You'd then generalize it.  

An analytical solution might be a cool project for the [Annals of Improbable Research](https://www.improbable.com/).

Hell, this might even be useful.  Who (among the people still reading) hasn't tried to fit a GLM to data that looks left-skewed when you log it?


```python
! jupyter nbconvert --to markdown README.ipynb
```

    [NbConvertApp] Converting notebook README.ipynb to markdown
    [NbConvertApp] Support files will be in README_files/
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Making directory README_files
    [NbConvertApp] Writing 20617 bytes to README.md



```python

```
