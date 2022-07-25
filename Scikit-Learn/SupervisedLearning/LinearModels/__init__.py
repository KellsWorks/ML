# Created by Kelvin Chidothi
# July 25, 2022 11:22

# LinearRegression fits a linear model with coefficients
# to minimize the residual sum of squares between the observed targets in the dataset,
# and the targets predicted by the linear approximation.
from sklearn import linear_model
import matplotlib.pyplot as plt


def main():
    linearModel = linear_model.LinearRegression()
    linearModel.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    var = linearModel.coef_  # Best fit line of the above coefficients

    plt.plot(var)

    print(var)


class OrdinaryLeastSquares:
    def __init(self):
        self.__init()
