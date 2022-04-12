Interactive demo for 3D Joint-Probability Density Distributions
--------------------------

Introduction
--------------------------

This is an example of using joint probability disributions (estimated with KDEs) to model a process and subsequently score and classify a new observation.
The underlying process is simulated by a 3D Gaussian mixture. The application visualises:
    * the raw data
    * the 3D kernel density estimate
    * the marginalised 3D density estimates (x, y and z)
    * the health envelope (equivalent to the lowest-value density contour)
    * the health scoring function
For a new observation controlled by the user, it also shows:
    * the position in 1D and 3D
    * the health score
    * the health status classification (healthy /unhealthy)

The user may interactively adjust:
    * x and y values of the point being evaluated
    * classification threshold (alpha)
    * number of contours drawn.

Note - There may be some delays in the interactiveness due to large amount of rendering needed for the 3D scatter plot. 

Using the application
--------------------------
Run ``main.py`` - it will host a local app web server at `<http://127.0.0.1:8050/>`_
(a link will be printed provide in your python console)

1. Investigate how the health score varies across the density distribution:
    * Use the tag sliders to change the current position and observe the resulting health score.
    * Observe that the health score = 1.0 (100%) at the peaks of the density function and tends toward 0.0 at the extremes.
    * The colour of the current position will change from green to red when it is outside of the health envelope (the red colored contour).

2. Investigate how alpha affects the health envelope:
    * Change the alpha value (health) threshold and observe how it affects the health envelope.
    * A value of 0.05 corresponds to 5% probability of observing a sample with lower health.


Troubleshooting
-------------------------
On a mac system you may encounter a ``ValueError: unknown locale:UTF-8`` during import of
processhealth.core.logging. If that happens, add the following commands to your ~/.bash_profile
or ~/.zshrc, and restart the terminal. If using PyCharm, you may need to restart the application:

    * ``export LC_ALL=en_US.UTF-8``
    * ``export LANG=en_US.UTF-8``
    * ``export LC_CTYPE=en_US.UTF-8``

Failure to kill server address `<http://127.0.0.1:8050/>`_ before re-running main.py will raise an
``OSError: [Errno 48] Address already in use`` error. If that happens, run the following commands to kill the server:

    * ``sudo lsof -i:8050``
    * ``kill <server PID>``