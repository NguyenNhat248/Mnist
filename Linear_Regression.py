import mlflow
import os
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.datasets import make_regression
import joblib
import datetime
from scipy.stats import zscore
from mlflow.tracking import MlflowClient 
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

DAGSHUB_USERNAME = "NguyenNhat248"  # Thay bằng username của bạn
DAGSHUB_REPO_NAME = "Mnist"
DAGSHUB_TOKEN = "4dd0f9a2823d65298c4840f778a4090d794b30d5"  # Thay bằng Access Token của bạn

def ly_thuyet_mlinear():
    # Tiêu đề chính với biểu tượng nổi bật
    st.title("📊 Hồi quy tuyến tính đa biến & Hồi quy đa thức")

    # Chia thành 2 phần
    tab1, tab2 = st.tabs(["📈 Hồi quy tuyến tính đa biến", "📉 Hồi quy đa thức"])

    with tab1:
        st.subheader("🔹 Hồi quy tuyến tính đa biến (Multiple Linear Regression - MLR)")

        # Hiển thị lý thuyết bên trái và công thức bên phải
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.markdown("""
            **Khái niệm:**  
            Hồi quy tuyến tính đa biến mở rộng từ hồi quy tuyến tính đơn biến, sử dụng nhiều biến độc lập để dự đoán biến phụ thuộc.
            """)
            st.write("📌 Công thức tổng quát:")
        
        with col2:
            st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n + \epsilon")

        # Danh sách giả định của mô hình
        st.markdown("### ✅ Giả định của MLR")
        st.markdown("""
        - **Tuyến tính**: Mối quan hệ giữa biến độc lập và biến phụ thuộc là tuyến tính.
        - **Không đa cộng tuyến**: Các biến độc lập không phụ thuộc mạnh vào nhau.
        - **Sai số phân phối chuẩn**: Sai số có phân phối chuẩn với trung bình bằng 0.
        """)

        # Công thức ước lượng hệ số
        st.write("📊 **Ước lượng tham số bằng phương pháp OLS:**")
        st.latex(r"\beta = (X^T X)^{-1} X^T Y")

        # Minh họa hồi quy tuyến tính đa biến
        st.write("### 📈 Minh họa MLR:")
        X = np.linspace(0, 10, 100)
        Y = 2 + 3 * X + np.random.randn(100) * 2
        fig, ax = plt.subplots()
        ax.scatter(X, Y, label="Dữ liệu thực tế")
        ax.plot(X, 2 + 3 * X, color='red', label="Đường hồi quy")
        ax.set_xlabel("X - Biến độc lập")
        ax.set_ylabel("Y - Biến phụ thuộc")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.subheader("🔹 Hồi quy đa thức (Polynomial Regression)")

        st.markdown("""
        **Khái niệm:**  
        Hồi quy đa thức là một dạng mở rộng của hồi quy tuyến tính, trong đó biến độc lập được nâng lên lũy thừa để phù hợp với dữ liệu phi tuyến.
        """)

        st.write("📌 Công thức hồi quy đa thức:")
        st.latex(r"y = w_0 + w_1 x + w_2 x^2 + ... + w_d x^d + \epsilon")

        # Minh họa hồi quy đa thức
        st.write("### 📉 Minh họa Hồi quy đa thức:")
        X_poly = np.linspace(-3, 3, 100)
        Y_poly = 2 + 1.5 * X_poly - 0.5 * X_poly**2 + np.random.randn(100) * 2
        fig_poly, ax_poly = plt.subplots()
        ax_poly.scatter(X_poly, Y_poly, label="Dữ liệu thực tế")
        ax_poly.plot(X_poly, 2 + 1.5 * X_poly - 0.5 * X_poly**2, color='green', label="Đường hồi quy đa thức")
        ax_poly.set_xlabel("X - Biến độc lập")
        ax_poly.set_ylabel("Y - Biến phụ thuộc")
        ax_poly.legend()
        st.pyplot(fig_poly)


## ----------------------------------- UPLOAD_DB ------------------------------
def up_load_db():
    st.header("Phân tích và xử lý dữ liệu")
    
    # 📥 Tải dữ liệu
    with st.expander("📥 Tải dữ liệu", expanded=True):
        uploaded_file = st.file_uploader("Tải file CSV (Titanic dataset)", type=["csv"])
        if uploaded_file is not None:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write("Dữ liệu đã được tải lên:")
            st.write(st.session_state.df.head(10))
            st.session_state.data_loaded = True

    # 🔍 Kiểm tra dữ liệu
    with st.expander("🔍 Kiểm tra dữ liệu"):
        if st.session_state.get("data_loaded", False):
            df = st.session_state.df

            # Tính số lượng giá trị thiếu
            missing_values = df.isnull().sum()

            # Xác định outliers bằng Z-score
            outlier_count = {
                col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
                for col in df.select_dtypes(include=['number']).columns
            }

            # Tạo báo cáo lỗi
            error_report = pd.DataFrame({
                "Cột": df.columns,
                "Giá trị thiếu": missing_values.values,
                "Outlier": [outlier_count.get(col, 0) for col in df.columns]
            })

            # Hiển thị bảng báo cáo với chiều rộng tối đa
            st.write("**Giá trị thiếu và Outlier:**")
            st.table(error_report)
        else:
            st.warning("Vui lòng tải dữ liệu trước.")

    # ⚙️ Xử lý dữ liệu
    with st.expander("⚙️ Xử lý dữ liệu"):
        if st.session_state.get("data_loaded", False):
            df = st.session_state.df.copy()

            # Loại bỏ cột
            dropped_cols = st.multiselect("Chọn cột cần loại bỏ:", df.columns.tolist(), default=["PassengerId", "Name", "Ticket", "Cabin"])
            df.drop(columns=dropped_cols, errors='ignore', inplace=True)
            st.write(f"Đã loại bỏ các cột: {', '.join(dropped_cols)}")

            # Điền giá trị thiếu
            st.write("Điền giá trị thiếu:")
            fill_missing_cols = st.multiselect("Chọn cột để điền giá trị thiếu:", df.columns.tolist())
            for col in fill_missing_cols:
                if df[col].isnull().any():
                    method = st.selectbox(f"Phương pháp điền cho cột {col}:", 
                                          options=["Median", "Mean", "Loại bỏ"], 
                                          key=f"fill_{col}")
                    if df[col].dtype in ['float64', 'int64']:
                        if method == "Trung vị (median)":
                            df[col].fillna(df[col].median(), inplace=True)
                        elif method == "Trung bình (mean)":
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif method == "Loại bỏ":
                            df.dropna(subset=[col], inplace=True)
                    else:
                        if method == "Mode":
                            df[col].fillna(df[col].mode()[0], inplace=True)
                        elif method == "Loại bỏ":
                            df.dropna(subset=[col], inplace=True)

            # Mã hóa biến phân loại
            st.write("Mã hóa các biến phân loại:")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            cols_to_encode = st.multiselect("Chọn cột để mã hóa:", categorical_cols)
            for col in cols_to_encode:
                df[col] = df[col].astype('category').cat.codes
                st.write(f"Đã mã hóa cột {col}.")

            # Chuẩn hóa dữ liệu số
            st.write("Chuẩn hóa dữ liệu số:")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if "Survived" in numeric_cols:
                numeric_cols.remove("Survived")  # Không chuẩn hóa cột nhãn

            if numeric_cols:  # Kiểm tra nếu có cột số để chuẩn hóa
                norm_method = st.selectbox("Chọn phương pháp chuẩn hóa:", ["Min-Max Scaling", "Standard Scaling"], key="norm_method")
                if norm_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                else:
                    scaler = StandardScaler()

                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.write(f"Đã chuẩn hóa các cột số: {', '.join(numeric_cols)}")

            # Cập nhật lại dữ liệu
            st.session_state.df = df
            st.session_state.data_processed = True

            # Hiển thị kết quả sau xử lý
            st.success("Dữ liệu đã được xử lý!")
            st.write(df.head(10))
        else:
            st.warning("Vui lòng tải dữ liệu trước.")


## -----------------------------------  LOGIN ACCOUNT MLFLOW  ------------------------------


def mlflow_input():
    """Thiết lập kết nối với MLflow trên DagsHub."""
    DAGSHUB_USERNAME = "NguyenNhat248"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "Mnist"
    DAGSHUB_TOKEN = "4dd0f9a2823d65298c4840f778a4090d794b30d5"  # Thay bằng Access Token của bạn

    # Đặt URI của MLflow trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng cách đặt biến môi trường
    import os
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm
    mlflow.set_experiment("Linear Regressions")

    # Lưu link MLflow vào session_state để dùng sau
    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"


## ----------------------------------- SPLIT SIZE DATASET ------------------------------
def chia_du_lieu():
    st.title("📊 Chia dữ liệu Train - Validation - Test")

    # Giới thiệu quy trình chia dữ liệu
    st.markdown("""
    **🔹 Tại sao cần chia dữ liệu?**  
    Chia dữ liệu giúp kiểm tra tính tổng quát của mô hình:
    - **Train (70%)**: Huấn luyện mô hình.
    - **Validation (15%)**: Tinh chỉnh tham số.
    - **Test (15%)**: Đánh giá cuối cùng.
    """)

    # Kiểm tra dữ liệu
    if "df" not in st.session_state:
        st.error("❌ Dữ liệu chưa được tải lên! Vui lòng tải dữ liệu trước.")
        st.stop()

    df = st.session_state.df  # Lấy dữ liệu từ session_state

    # Tùy chỉnh tỷ lệ chia dữ liệu
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20, step=5)
    with col2:
        val_size = st.slider("📌 Chọn % dữ liệu Validation", 0, 50, 15, step=5)

    remaining_size = 100 - test_size
    train_size = remaining_size - val_size

    st.write(f"📊 **Tỷ lệ phân chia:** Train={train_size}%, Validation={val_size}%, Test={test_size}%")

    # Hiển thị tiến trình chia dữ liệu
    if st.button("✅ Xác nhận Chia"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Chia dữ liệu theo tỷ lệ đã chọn
        progress_bar.progress(30)
        progress_text.text("🔄 Đang chia tập Test...")
        time.sleep(0.5)
        train_full, test = train_test_split(df, test_size=test_size / 100, random_state=42)

        progress_bar.progress(70)
        progress_text.text("🔄 Đang chia tập Train và Validation...")
        time.sleep(0.5)
        train, val = train_test_split(train_full, test_size=val_size / (100 - test_size), random_state=42)

        # Lưu dữ liệu vào session_state
        st.session_state.train = train
        st.session_state.test = test
        st.session_state.val = val

        # Hiển thị số lượng mẫu
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [train.shape[0], val.shape[0], test.shape[0]]
        })
        st.table(summary_df)

        # Biểu đồ minh họa tỷ lệ dữ liệu
        labels = ["Train", "Validation", "Test"]
        sizes = [train_size, val_size, test_size]
        colors = ["#2E86C1", "#F1C40F", "#E74C3C"]

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={"edgecolor": "white"})
        ax.axis("equal")  # Hiển thị pie chart theo hình tròn

        st.pyplot(fig)

        # Hoàn thành tiến trình
        progress_bar.progress(100)
        progress_text.text("✅ Chia dữ liệu hoàn thành!")
        st.success("🎉 Dữ liệu đã được chia thành công!")



def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức bằng Gradient Descent kèm hiển thị tiến trình & biểu đồ lỗi."""
    
    st.subheader("🎯 Huấn luyện hồi quy đa thức")
    
    # Chuyển đổi dữ liệu nếu là Pandas
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức (không bao gồm tương tác)
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Thêm bias (x0 = 1)
    m, n = X_poly.shape
    X_b = np.c_[np.ones((m, 1)), X_poly]

    # Khởi tạo trọng số ngẫu nhiên
    w = np.random.randn(n + 1, 1) * 0.01

    # Gradient Descent
    loss_history = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)
        
        if np.isnan(gradients).any():
            st.error("❌ Gradient chứa giá trị NaN! Hãy kiểm tra dữ liệu hoặc learning rate.")
            return None
        
        w -= learning_rate * gradients
        
        # Tính toán lỗi trung bình bình phương (MSE) để hiển thị biểu đồ
        mse = np.mean((X_b.dot(w) - y_train) ** 2)
        loss_history.append(mse)

        if iteration % 50 == 0:
            progress_bar.progress(iteration / n_iterations)
            progress_text.text(f"🔄 Đang huấn luyện... ({iteration}/{n_iterations})")

    progress_bar.progress(100)
    progress_text.text("✅ Huấn luyện hoàn tất!")

    # Hiển thị biểu đồ lỗi giảm dần
    st.write("📉 **Biểu đồ MSE qua các vòng lặp:**")
    fig, ax = plt.subplots()
    ax.plot(range(n_iterations), loss_history, color="red")
    ax.set_xlabel("Số vòng lặp")
    ax.set_ylabel("MSE")
    ax.set_title("Quá trình giảm lỗi trong Gradient Descent")
    st.pyplot(fig)

    return w


def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent với tiến trình & biểu đồ lỗi."""

    st.subheader("📈 Huấn luyện hồi quy tuyến tính bội")

    # Chuyển đổi dữ liệu
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra dữ liệu có NaN hoặc Inf không
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        st.error("❌ Dữ liệu chứa NaN! Hãy kiểm tra lại dữ liệu đầu vào.")
        return None
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        st.error("❌ Dữ liệu chứa giá trị vô cùng! Hãy kiểm tra lại.")
        return None

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Thêm bias
    m, n = X_train.shape
    X_b = np.c_[np.ones((m, 1)), X_train]

    # Khởi tạo trọng số
    w = np.random.randn(n + 1, 1) * 0.01

    # Gradient Descent
    loss_history = []
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        if np.isnan(gradients).any():
            st.error("❌ Gradient chứa giá trị NaN! Hãy kiểm tra learning rate.")
            return None

        w -= learning_rate * gradients

        # Tính toán lỗi
        mse = np.mean((X_b.dot(w) - y_train) ** 2)
        loss_history.append(mse)

        if iteration % 50 == 0:
            progress_bar.progress(iteration / n_iterations)
            progress_text.text(f"🔄 Đang huấn luyện... ({iteration}/{n_iterations})")

    progress_bar.progress(100)
    progress_text.text("✅ Huấn luyện hoàn tất!")

    # Hiển thị biểu đồ lỗi
    st.write("📉 **Biểu đồ MSE qua các vòng lặp:**")
    fig, ax = plt.subplots()
    ax.plot(range(n_iterations), loss_history, color="blue")
    ax.set_xlabel("Số vòng lặp")
    ax.set_ylabel("MSE")
    ax.set_title("Quá trình giảm lỗi trong Gradient Descent")
    st.pyplot(fig)

    return w

def train_and_log_model():
    """Giao diện Streamlit để huấn luyện mô hình hồi quy và log kết quả lên MLflow."""
    
    st.title("📈 Huấn luyện mô hình Hồi quy")

    # Thiết lập kết nối với MLflow
    mlflow_input()

    # Chọn loại mô hình
    model_type = st.selectbox("🔍 Chọn mô hình:", ["Hồi quy tuyến tính bội", "Hồi quy đa thức"])
    
    # Nếu chọn hồi quy đa thức, cho phép chọn bậc của đa thức
    degree = 2
    if model_type == "Hồi quy đa thức":
        degree = st.slider("🎚 Chọn bậc của đa thức:", min_value=2, max_value=5, value=2)
    
    # Chọn hyperparameters
    learning_rate = st.number_input("⚡ Learning rate:", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
    n_iterations = st.number_input("🔄 Số lần lặp:", min_value=100, max_value=1000, value=200, step=50)
    n_splits = st.number_input("📊 Số folds cho KFold Cross-Validation:", min_value=2, max_value=10, value=5, step=1)
    
    # Nút để huấn luyện mô hình
    if st.button("🚀 Huấn luyện mô hình"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        st.write("🔄 Đang huấn luyện mô hình...")

        # Tạo dữ liệu giả lập để huấn luyện
        X, y = make_regression(n_samples=100, n_features=2, noise=0.1)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        mse_scores, rmse_scores, r2_scores = [], [], []
        
        with mlflow.start_run() as run:
            for fold_idx, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Chuẩn hóa dữ liệu
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Chọn mô hình
                if model_type == "Hồi quy đa thức":
                    model = train_polynomial_regression(X_train_scaled, y_train, degree, learning_rate, n_iterations)
                else:
                    model = train_multiple_linear_regression(X_train_scaled, y_train, learning_rate, n_iterations)

                # Thêm bias vào X_test
                if model_type == "Hồi quy đa thức":
                    X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
                    X_b_test = np.c_[np.ones((X_test.shape[0], 1)), X_test_poly]
                else:
                    X_b_test = np.c_[np.ones((X_test.shape[0], 1)), X_test_scaled]

                # Dự đoán giá trị
                y_pred = X_b_test.dot(model)

                # Tính metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                mse_scores.append(mse)
                rmse_scores.append(rmse)
                r2_scores.append(r2)

                # Cập nhật tiến trình
                progress = int(((fold_idx + 1) / n_splits) * 100)
                progress_bar.progress(progress)
                progress_text.text(f"🔄 Tiến trình: {progress}% - Huấn luyện Fold {fold_idx + 1}/{n_splits}")

                time.sleep(0.5)  # Giả lập thời gian chờ huấn luyện

            # Log giá trị trung bình của các metrics lên MLflow
            mlflow.log_metrics({
                "MSE": np.mean(mse_scores),
                "RMSE": np.mean(rmse_scores),
                "R2_score": np.mean(r2_scores)
            })

            # Lưu mô hình
            model_dir = "saved_models"
            os.makedirs(model_dir, exist_ok=True)
            model_filename = f"{model_dir}/{model_type.replace(' ', '_').lower()}_model.pkl"
            joblib.dump(model, model_filename)

            # Log mô hình lên MLflow
            mlflow.log_artifact(model_filename)

            # Lấy link MLflow
            run_id = run.info.run_id
            mlflow_run_url = f"{st.session_state['mlflow_url']}/#/experiments/0/runs/{run_id}"
            
            # ✅ Hiển thị kết quả trên Streamlit
            st.subheader("📊 Kết quả huấn luyện mô hình")
            st.write(f"✅ Mô hình đã được lưu tại: `{model_filename}`")
            st.markdown(f"🔗 **Tải mô hình:** [Download {model_filename}](./{model_filename})")
            st.markdown(f"🔗 **MLflow Tracking:** [Xem kết quả trên MLflow]({mlflow_run_url})")
            
            progress_bar.progress(100)
            progress_text.text("✅ Tiến trình: 100% - Hoàn thành huấn luyện!")
            st.success("✅ Huấn luyện và logging hoàn tất!")


##------------------------------------ TRACKING MLFLOW -------------------------------
def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def display_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow với thanh trạng thái tiến trình."""
    st.title("📊 MLflow Experiment Viewer")

    # Lấy danh sách thí nghiệm
    experiment_name = "Linear Regressions"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # --- 🏃‍♂️ Lấy danh sách Runs với thanh trạng thái ---
    st.write("### 🔄 Đang tải danh sách Runs...")
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    total_runs = len(runs)
    run_info = []
    
    progress_bar = st.progress(0)  # Thanh tiến trình

    for i, (_, run) in enumerate(runs.iterrows()):
        run_id = run["run_id"]
        run_data = mlflow.get_run(run_id)
        run_tags = run_data.data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Tên Run
        created_time = format_time_relative(run_data.info.start_time)
        duration = (run_data.info.end_time - run_data.info.start_time) / 1000 if run_data.info.end_time else "Đang chạy"
        source = run_tags.get("mlflow.source.name", "Unknown")

        run_info.append({
            "Run Name": run_name,
            "Run ID": run_id,
            "Created": created_time,
            "Duration (s)": duration if isinstance(duration, str) else f"{duration:.1f}s",
            "Source": source
        })

        # Cập nhật thanh tiến trình
        progress_bar.progress(int((i + 1) / total_runs * 100))

    progress_bar.empty()  # Xóa thanh tiến trình khi hoàn thành

    # Sắp xếp và hiển thị bảng danh sách Runs
    run_info_df = pd.DataFrame(run_info).sort_values(by="Created", ascending=False)
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_info_df, use_container_width=True)

    # Chọn Run từ dropdown
    run_names = run_info_df["Run Name"].tolist()
    selected_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_names)

    # Lấy Run ID tương ứng
    selected_run_id = run_info_df.loc[run_info_df["Run Name"] == selected_run_name, "Run ID"].values[0]
    selected_run = mlflow.get_run(selected_run_id)

    # --- 📝 ĐỔI TÊN RUN ---
    st.write("### ✏️ Đổi tên Run")
    new_run_name = st.text_input("Nhập tên mới:", selected_run_name)
    if st.button("💾 Lưu tên mới"):
        try:
            mlflow.set_tag(selected_run_id, "mlflow.runName", new_run_name)
            st.success(f"✅ Đã đổi tên thành **{new_run_name}**. Hãy tải lại trang để thấy thay đổi!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")

    # --- 🗑️ XÓA RUN ---
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(selected_run_id)
            st.success(f"✅ Đã xóa run **{selected_run_name}**! Hãy tải lại trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")

    # --- HIỂN THỊ CHI TIẾT RUN ---
    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")

        start_time_ms = selected_run.info.start_time
        start_time = datetime.datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không có thông tin"
        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Hiển thị model artifact (nếu có)
        model_artifact_path = f"{st.session_state['mlflow_url']}/{selected_experiment.experiment_id}/{selected_run_id}/artifacts/model"
        st.write("### 📂 Model Artifact:")
        st.write(f"📥 [Tải mô hình]({model_artifact_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

def predict_survival():
    df = pd.read_csv("data1.csv")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    df = df[features + ["Survived"]]

    # Xử lý dữ liệu thiếu
    imputer = SimpleImputer(strategy="median")
    df["Age"] = imputer.fit_transform(df[["Age"]])
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Chuyển đổi dữ liệu dạng chữ thành số
    label_encoders = {}
    for col in ["Sex", "Embarked"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Huấn luyện mô hình Polynomial Regression bậc 2
    X = df[features]
    y = df["Survived"]
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Giao diện Streamlit
    st.title("Dự đoán khả năng sống sót trên Titanic")

    pclass = st.selectbox("Hạng vé (Pclass)", [1, 2, 3])
    sex = st.selectbox("Giới tính", ["male", "female"])
    age = st.number_input("Tuổi", min_value=0, max_value=100, value=30)
    sibsp = st.number_input("Số anh chị em / vợ chồng đi cùng (SibSp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Số cha mẹ / con cái đi cùng (Parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Giá vé (Fare)", min_value=0.0, max_value=600.0, value=30.0)
    embarked = st.selectbox("Cảng đi (Embarked)", ["C", "Q", "S"])

    if st.button("Dự đoán"):
        input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=features)
        input_data["Sex"] = label_encoders["Sex"].transform([sex])[0]
        input_data["Embarked"] = label_encoders["Embarked"].transform([embarked])[0]
        
        input_data_poly = poly.transform(input_data)
        prediction = model.predict(input_data_poly)[0]
        survival = "Sống sót" if prediction >= 0.5 else "Không sống sót"
        
        st.success(f"Dự đoán: {survival}")
        if prediction >= 0.5:
            st.write("Quyết định cuối cùng: Hành khách này có khả năng sống sót.")
        else:
            st.write("Quyết định cuối cùng: Hành khách này không có khả năng sống sót.")
        
        # Kiểm tra xem hành khách này có trong tập dữ liệu gốc không
        matched = df[
            (df["Pclass"] == pclass) &
            (df["Sex"] == label_encoders["Sex"].transform([sex])[0]) &
            (df["Age"] == age) &
            (df["SibSp"] == sibsp) &
            (df["Parch"] == parch) &
            (df["Fare"] == fare) &
            (df["Embarked"] == label_encoders["Embarked"].transform([embarked])[0])
        ]
        
        if not matched.empty:
            actual_survival = "Sống sót" if matched["Survived"].values[0] == 1 else "Không sống sót"
            st.info(f"Thông tin trùng khớp trong tập dữ liệu! Thực tế: {actual_survival}")
        else:
            st.info("Hành khách này không có trong tập dữ liệu gốc.")



def mt_Regression():
  
    # Thiết lập CSS để hỗ trợ hiển thị tabs với hiệu ứng hover và thanh cuộn
    st.markdown(
        """
        <style>
        .stTabs [role="tablist"] {
            overflow-x: auto;
            white-space: nowrap;
            display: flex;
            scrollbar-width: thin;
            scrollbar-color: #888 #f0f0f0;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar {
            height: 6px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-thumb {
            background-color: #888;
            border-radius: 3px;
        }
        .stTabs [role="tablist"]::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        .stTabs [role="tab"]:hover {
            background-color: #f0f0f0;
            transition: background-color 0.3s ease-in-out;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🖊️ Multiple & Polynomial Regression App")

    # Ensure the tab names are properly separated
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📘 Lý thuyết MLR & Polynomial Regression", 
    "📥 Tải & tiền xử lý dữ liệu",  
    "🔀 Chia dữ liệu", 
    "🤖 Huấn luyện mô hình", 
    "🔍 Thông tin huấn luyện",
    "🧠 Dự đoán"
    ])

    with tab1: 
        ly_thuyet_mlinear() 
    with tab2: 
        up_load_db() 
    with tab3: 
        chia_du_lieu() 
    with tab4:
        train_and_log_model() 
    with tab5:
        display_mlflow_experiments() 
    with tab6: 
        predict_survival()

def run():
    mt_Regression()

if __name__ == "__main__": 
    run()

