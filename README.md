# Calculate Curvature
The simple idea is this: I find out, roughly, the magnitude of the second derivative. I do thatby taking a REALLY small step. When I know it, I figure out how big of a step to take. A high second derivative means that the slope is increasing, which since we're going in away from the gradient, means that it's getting shallower. That's bad! Negative means it'll get steeper. That's good!

### Taylor expansion:
f(x+d) = f(x) + d*f'(x) + d^2*f''(x)/2

### In this case: 
It's a little tricky to think about the distance (dx) in the multidimensional case. But what it works out to is that you go a distance "LR * ||f'||", in the direction of f'. And, your slope is the magnitude of the derivative, which is ||f'||. So, that makes the terms a little easier to parse.

___
f(x+d) = f(x) + LR* || f' ||<sup>2</sup> + (LR<sup>2</sup>|| f' ||<sup>2</sup>) * f'' / 2
___

If only this was important enough to break out the latex.

__ANYWAYS...__

We care about the difference between f(x+d) and f(x). Let's say `f(x+d) - f(x) = DIFF`

Moving things around a little gives you this:

___
f'' = 2 * (DIFF - LR* || f' ||<sup>2</sup>) / ((LR<sup>2</sup>|| f' ||<sup>2</sup>))
___

__Huzzah!__

### How to use it
Now that we have the second derivative, we use it to modify the step size. You get your learning rate by 

___
##### TRUE_LR = sigmoid(-f'')
___

And you update from the original point, in the direction of the gradient, that much. Not so tough!

