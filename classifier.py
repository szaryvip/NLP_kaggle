from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from bag_of_words import all_as_numbers, is_bad, divide_data, sentence_as_number
from lemmatization.lemmatization import lemmatize_all_csv

words = lemmatize_all_csv('data/train.csv')

x = all_as_numbers('data/train.csv', words)
y = is_bad('data/train.csv')

X_train, X_test, y_train, y_test = divide_data(x, y, 0.8)


scaler = StandardScaler()
scaler.fit(X_train)
X_train_norm = scaler.transform(X_train)
X_test_norm = scaler.transform(X_test)
print("jestem w sieci")
# Sieć neuronowa

neural_network = MLPClassifier(hidden_layer_sizes=(100, 100)
                              ,batch_size=2
                              ,learning_rate_init=0.1
                              ,max_iter=10
                              ,verbose=True)

neural_network.fit(X_train_norm, y_train)  # trening

prediction = neural_network.predict_proba(X_train_norm)[:,1]  # predykcja

print(classification_report(y_train, (prediction > 0.5).astype(int)))  # train

prediction = neural_network.predict_proba(X_test_norm)[:,1]  # predykcja

print(classification_report(y_test, (prediction > 0.5).astype(int)))  # test

# Testy
zdanie = "#DEAD #TNT #SWORD Kill every creeper and use X-RAY !!1!one!"

zdanie_norm = sentence_as_number(zdanie, words)
zdanie_norm = scaler.transform(zdanie_norm)

print(neural_network.predict_proba(zdanie_norm)) # (będzie 0, będzie 1)
