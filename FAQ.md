# FAQ

* Is this competition really so cool?
Yes.

## Known issues

* __render('human') and visualization=True render shadows also in the robot camera__  
If you use `env.render('human')` or put `visualization=True` in the `evaluate` function it will open a GUI showing the environment. The GUI renders the environment using shadows as well.
However, as a side effect, using the GUI will also make the environment render the shadows on the robot camera.  
_This means that robots trained on images with the GUI on might have a different performance than when the GUI is off, since images will be different._
