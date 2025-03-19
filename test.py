import joblib

# Giả sử bạn có một mô hình SVM đã huấn luyện
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Load dữ liệu MNIST nhỏ
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Huấn luyện mô hình SVM
model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)

# Lưu mô hình
joblib.dump(model, "svm_mnist_linear.joblib")
print("Mô hình đã được lưu thành công!")
model_path = "svm_mnist_poly.joblib"  # Đổi tên thành file mô hình có sẵn
