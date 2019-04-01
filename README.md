# Bayesian Hilbert Maps
## Online Bayesian Hilbert Mapping
Hilbert occupancy mapping without tuning regularization parameters

**Tutorials**
[BHM_tutorial.ipynb](BHM_tutorial.ipynb)

**Demonstrations**
Now BHM is available in both numpy and pytorch. A few fixes to by done with the gpu version.  

<img src="Outputs/intel.gif" width="600">

**Videos:**
[https://youtu.be/LDrLsvfJ0V0](https://youtu.be/LDrLsvfJ0V0)

[https://youtu.be/gxi0JKuzJvU](https://youtu.be/gxi0JKuzJvU)

[https://youtu.be/iNXnRjLEsHQ](https://youtu.be/iNXnRjLEsHQ)

**Example:**
```python
import sbhm

X = #numpy array of size (N,2)
y = #numpy array of size (N,)
X_pred = #numpy array of size (N_pred,2)

model = sbhm.SBHM(gamma)
model.fit(X, y)
y_pred = model.predict(X_pred)

# with pytorch
TBD
```

**Papers:**
Introduction to Bayesian Hilbert Maps:
```
@inproceedings{senanayake2017bayesian,
  title={Bayesian hilbert maps for dynamic continuous occupancy mapping},
  author={Senanayake, Ransalu and Ramos, Fabio},
  booktitle={Conference on Robot Learning},
  pages={458--471},
  year={2017}
}
```

Examples with moving robots and the similarities to Gaussian process based techniques:
```
@inproceedings{senanayake2018continuous,
  title={Building Continuous Occupancy Maps with Moving Robots},
  author={Senanayake, Ransalu and Ramos, Fabio},
  booktitle={Proceedings of the Thirty Second AAAI Conference on Artificial Intelligence},
  year={2018},
  organization={AAAI Press}
}
```

Learning hinge points and kernel parameters:
```
@inproceedings{senanayake2018automorphing,
  title={Automorphing Kernels for Nonstationarity in Mapping Unstructured Environments},
  author={Senanayake*, Ransalu and Tomkins*, Anthony and Ramos, Fabio},
  booktitle={Conference on Robot Learning},
  pages={--},
  year={2018}
}
```
code: [https://github.com/MushroomHunting/autormorphing-kernels](https://github.com/MushroomHunting/autormorphing-kernels)

Fast fusion with multiple robots
```
@inproceedings{zhi2019fusion,
  title={Continuous Occupancy Map Fusion with Fast Bayesian Hilbert Maps},
  author={Zhi, William and Ott, Lionel and Senanayake, Ransalu and Ramos, Fabio},
  booktitle={The International Conference on Robotics and Automation (ICRA)},
  pages={--},
  year={2019}
}
```

