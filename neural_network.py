class Neural_Network(object):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """Initialization of the neural network parameters"""
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))                    
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))                    
      
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_prime(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))   #Numerical stability trick
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.output = self.softmax(self.z2)
        return self.output
    
    def compute_loss(self, y_true, y_pred):
        """Cross-entropy loss computation"""
        correct_probs = np.sum(y_true*y_pred, axis = 1)  #Probability of the correct class
        loss = -np.mean(np.log(correct_probs + 1e-15))
        return loss
    
    def backward(self, X, y_true, y_pred):
        """Backward pass"""
        N = X.shape[0]
        delta2 = (y_pred - y_true)/N
        
        self.dw2 = np.dot(self.a1.T, delta2)
        self.db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = np.dot(delta2, self.w2.T)*self.relu_prime(self.z1)
        
        self.dw1 = np.dot(X.T, delta1)
        self.db1 = np.sum(delta1, axis=0, keepdims=True)
        
    def update_parameters(self):
        """Weights and bias update"""
        self.w1 -= self.learning_rate*self.dw1
        self.b1 -= self.learning_rate*self.db1
        self.w2 -= self.learning_rate*self.dw2
        self.b2 -= self.learning_rate*self.db2
        
    def train(self, X_train, y_train, epochs):
        """Training loop"""
        loss_tab = []
        for epoch in range(epochs):
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_train, y_pred)
            self.backward(X_train, y_train, y_pred)
            self.update_parameters()
            loss_tab.append(loss)
        return loss_tab
        
    def predict(self, X):
        """Prediction function"""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
