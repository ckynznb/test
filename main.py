import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
X=datasets.load_iris()['data']
Y=datasets.load_iris()['target']
Y[Y>1]=1
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,stratify=Y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cal_grad(y, t):
    grad = np.sum(t - y) / t.shape[0]
    return grad

def cal_cross_loss(y, t):
    loss=np.sum(-y * np.log(t)- (1 - y) * np.log(1 - t))/t.shape[0]
    return loss

class LR:
    def __init__(self, in_num, lr, iters, train_x, train_y, test_x, test_y):
        self.w = np.random.rand(in_num)
        self.b = np.random.rand(1)
        self.lr = lr
        self.iters = iters
        self.x = train_x
        self.y = train_y
        self.test_x=test_x
        self.test_y=test_y


    def forward(self, x):
        #也即
        self.a = np.dot(x, self.w) + self.b
        self.g = sigmoid(self.a)
        return self.g

    def backward(self, x, grad):
        w = grad * x
        b = grad
        self.w = self.w - self.lr * w
        self.b = self.b - self.lr * b

    def valid_loss(self):
        pred = sigmoid(np.dot(self.test_x, self.w) + self.b)
        return cal_cross_loss(self.test_y, pred)

    def train_loss(self):
        pred = sigmoid(np.dot(self.x, self.w) + self.b)
        return cal_cross_loss(self.y, pred)

    def train(self):
        for iter in range(self.iters):
            ##这里我采用随机梯度下降的方法

            for i in range(self.x.shape[0]):
                t = self.forward(self.x[i])
                grad = cal_grad(self.y[i], t)
                self.backward(self.x[i], grad)

            train_loss = self.train_loss()
            valid_loss = self.valid_loss()
            if iter%5==0:
                print("当前迭代次数为：", iter, "训练loss:", train_loss, "验证loss:", valid_loss)
w=[0.2,0.3]
x=[1,1]
y=1
print(cal_grad(y,x))