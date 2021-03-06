import numpy as np


class LinearRegression(object):
    def __init__(self, fit_intercept=True, copy_X=True):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X # 기존 x값 저장

        self._coef = None
        self._intercept = None
        self._new_X = None

    def fit(self, X, y):
        self._new_X = np.array(X) # list타입으로 올수있기때문에 np.array로 함
        y = y.reshape(-1,1) # 계산을 편하게 하기 위해 1 dimensional을 2 dimensional로 변경

        if self.fit_intercept: # fit_intercept란 절편(y=ax+b에서 b)
            intercept_vector = np.ones([len(self._new_X), -1]) # (300,1)의 intercept_vector를 만듬
            self._new_X = np.concatenate( # concatenate로 intercept_vector와 _new_X를 합침
                (intercept_vector, self._new_X), axis=1)

        # inv => 역함수 // (X^T * X)^(-1) * X^T * y => 2 by 1의 vector 생성
        # flatten()으로 1 demension으로 풀어줌
        weights = np.linalg.inv(self._new_X.T.dot(self._new_X)).dot(
                                                    self._new_X.T.dot(y)).flatten()

        if self.fit_intercept:
            self._intercept = weights[0] # w = [ w0 ]
            self._coef = weights[1:]     #       w1      w1 -> coef (계수)
        else:
            self._coef = weights
        """
        Linear regression 모델을 적합한다.
        Matrix X와 Vector Y가 입력 값으로 들어오면 Normal equation을 활용하여, weight값을
        찾는다. 이 때, instance가 생성될 때, fit_intercept 설정에 따라 fit 실행이 달라진다.
        fit을 할 때는 입력되는 X의 값은 반드시 새로운 변수(self._new_X)에 저장
        된 후 실행되어야 한다.
        fit_intercept가 True일 경우:
            - Matrix X의 0번째 Column에 값이 1인 column vector를추가한다.

        적합이 종료된 후 각 변수의 계수(coefficient 또는 weight값을 의미)는 self._coef와
        self._intercept_coef에 저장된다. 이때 self._coef는 numpy array을 각 변수항의
        weight값을 저장한 1차원 vector이며, self._intercept_coef는 상수항의 weight를
        저장한 scalar(float) 이다.
        Parameters
        ----------
        X : numpy array, 2차원 matrix 형태로 [n_samples,n_features] 구조를 가진다
        y : numpy array, 1차원 vector 형태로 [n_targets]의 구조를 가진다.

        Returns
        -------
        self : 현재의 인스턴스가 리턴된다
        """

    def predict(self, X):
        test_X = np.array(X)

        if self.fit_intercept:
            intercept_vectpr = np.ones([len(test_X), 1])
            test_X = np.concatenate((intercept_vector, test_X), axis=1)

            weights = np.concatenate(([self._intercept], self._coef), axis=0)
        else:
            weights = self._coef
        return test_X.dot(weights)
        """
        적합된 Linear regression 모델을 사용하여 입력된 Matrix X의 예측값을 반환한다.
        이 때, 입력된 Matrix X는 별도의 전처리가 없는 상태로 입력되는 걸로 가정한다.
        fit_intercept가 True일 경우:
            - Matrix X의 0번째 Column에 값이 1인 column vector를추가한다.
        normalize가 True일 경우:
            - Standard normalization으로 Matrix X의 column 0(상수)를 제외한 모든 값을
              정규화을 실행함
            - 정규화를 할때는 self._mu_X와 self._std_X 에 있는 값을 사용한다.
        Parameters
        ----------
        X : numpy array, 2차원 matrix 형태로 [n_samples,n_features] 구조를 가진다

        Returns
        -------
        y : numpy array, 예측된 값을 1차원 vector 형태로 [n_predicted_targets]의
            구조를 가진다.
        """

    @property
    def coef(self): # 계수값
        return self._coef

    @property
    def intercept(self):
        return self._intercept
