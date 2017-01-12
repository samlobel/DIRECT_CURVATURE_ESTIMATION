# Calculate Curvature

The simple idea is this: I find out, roughly, the magnitude of the second derivative. I do thatby taking a REALLY small step. When I know it, I figure out how big of a step to take.


Formula:

f(x+d) = f(x) + d*f'(x) + d^2*f''(x)/2

Say f(x+d) - f(x) = DIFF.

DIFF - d*f'(x) = d^2*f''(x)/2
2*(DIFF - d*f'(x))/d^2 = f''(x)

d^2 = LR^2*f'(x)^2



If you have a gradient of (2,1): you update by (0.2,0.1). I think that means that I was wrong
before. It should be that d=LR_ORIG. Wait, maybe not. You go a distance of LR*|f'|. I wish I could still think clearly....

Let's say you have x^2. Derivative 2x. Evaluate at 1. Derivative 2. Change by 0.2, goes to 1.2.
Becomes 1.44, but 1.4 in approx. That's 0.4 difference, which is LR*grad*grad. So, delta SHOULD be
LR*GRAD*GRAD. Anything different from that is the second derivative. 

The second derivative, in the direction of the gradient: When you go d, the derivative changes by f''*d, so the output chnages by f''*d^2/2

d^2 is LR*LR*|f'|^2

SO:
DIFF - d*f'(x) = d^2*f''(x)/2

I think I may be playing fast with math, but I'm not sure. I'm reducing it to one dimension by
picking my direction of gradient. And then, I say that my update direction is LR*|GRAD|. So, the amount the slope should change in the distance it goes is LR*|GRAD|*|f''| (where f'' is evaluated in the direction of the gradient). So the amount the total should change is LR^2*GRAD^2*f''.

Feeling pretty good today, let's redo it. 
If you have f(x,y) = 2x + 4y. The gradient is pretty clearly 2,4. That's because if you go a length of one in this direction:
What's one in this direction? It's (2,4) / sqrt(2^2+4^2). sqrt(20) = 2*sqrt(5). So you get (sqrt(5)/2, sqrt(5))



## HOW FAR SHOULD YOU GO?!
That's a question of how big the derivative is, AND how big the second derivative is. It's asking, how far would you have to go at that second derivative if you were to make things worse for this sample. And that's the time it takes for the second derivative to make the first derivative zero.

f'(x+d) = f'(x) + df''(x). Setting the first to zero,

d = -f' / f''  . So, let's say we go half that distance. That's probably a safe estimate of how far we should really go.

If we go distance D, the derivative changes by D*f''. 
D*f'' = f'/2



So, the first time, we update 0.0001. I just hope we don't run into problems with floating point division. We should do everythign in float64. 

The next thing is, how do you get the magnitude of d, and the magnitude of f'(x)? This stuff is way easier in one dimension. I think d is LR*f'(x). That makes things a little easier. 



