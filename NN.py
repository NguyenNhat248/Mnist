import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import openml
import os
import torch.nn as nn
import mlflow
import plotly.express as px
import shutil
import time
import random
from streamlit_drawable_canvas import st_canvas
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
from torchvision import transforms
from datetime import datetime
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlflow.tracking import MlflowClient
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 



def mlflow_input():

    DAGSHUB_USERNAME = "NguyenNhat248"  # Thay bằng username của bạn
    DAGSHUB_REPO_NAME = "Mnist"
    DAGSHUB_TOKEN = "4dd0f9a2823d65298c4840f778a4090d794b30d5"  # Thay bằng Access Token của bạn

    # Đặt URI MLflow để trỏ đến DagsHub
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow")

    # Thiết lập authentication bằng Access Token
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

    # Đặt thí nghiệm MLflow
    mlflow.set_experiment("Neural Network")   

    st.session_state['mlflow_url'] = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO_NAME}.mlflow"



def format_time_relative(timestamp_ms):
    """Chuyển timestamp milliseconds thành thời gian dễ đọc."""
    if timestamp_ms is None:
        return "N/A"
    dt = datetime.fromtimestamp(timestamp_ms / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def show_mlflow_experiments():
    """Hiển thị danh sách Runs trong MLflow với giao diện trực quan."""
    st.title("📊 Trình xem MLflow Experiments")
    
    # Nhập thông tin MLflow
    mlflow_input()
    
    # Xác định tên experiment
    exp_name = "Neural Network"
    exp_list = mlflow.search_experiments()
    chosen_exp = next((e for e in exp_list if e.name == exp_name), None)
    
    if not chosen_exp:
        st.error(f"❌ Experiment '{exp_name}' không tồn tại!")
        return
    
    # Hiển thị thông tin cơ bản
    st.subheader(f"📌 Experiment: {exp_name}")
    st.write(f"**Experiment ID:** {chosen_exp.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if chosen_exp.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Lưu trữ tại:** {chosen_exp.artifact_location}")
    
    # Truy xuất danh sách Runs
    runs_df = mlflow.search_runs(experiment_ids=[chosen_exp.experiment_id])
    if runs_df.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return
    
    # Xử lý dữ liệu Runs
    run_data = []
    for _, run in runs_df.iterrows():
        run_id = run["run_id"]
        run_details = mlflow.get_run(run_id)
        run_name = run_details.data.tags.get("mlflow.runName", f"Run {run_id[:8]}")
        created_time = format_time_relative(run_details.info.start_time)
        duration = (run_details.info.end_time - run_details.info.start_time) / 1000 if run_details.info.end_time else "Đang chạy"
        source = run_details.data.tags.get("mlflow.source.name", "Unknown")
        
        run_data.append({
            "Tên Run": run_name,
            "Run ID": run_id,
            "Tạo lúc": created_time,
            "Thời gian (s)": f"{duration:.1f}s" if isinstance(duration, float) else duration,
            "Nguồn": source
        })
    
    run_df = pd.DataFrame(run_data).sort_values(by="Tạo lúc", ascending=False)
    st.write("### 🏃‍♂️ Danh sách Runs:")
    st.dataframe(run_df, use_container_width=True)
    
    # Chọn Run cụ thể để xem
    run_choices = run_df["Tên Run"].tolist()
    chosen_run_name = st.selectbox("🔍 Chọn một Run để xem chi tiết:", run_choices)
    chosen_run_id = run_df.loc[run_df["Tên Run"] == chosen_run_name, "Run ID"].values[0]
    chosen_run = mlflow.get_run(chosen_run_id)
    
    # Chỉnh sửa tên Run
    st.write("### ✏️ Đổi tên Run")
    new_name = st.text_input("Nhập tên mới:", chosen_run_name)
    if st.button("💾 Lưu thay đổi"):
        try:
            mlflow.set_tag(chosen_run_id, "mlflow.runName", new_name)
            st.success(f"✅ Đã đổi tên thành **{new_name}**. Vui lòng tải lại trang!")
        except Exception as e:
            st.error(f"❌ Lỗi khi đổi tên: {e}")
    
    # Xóa Run
    st.write("### ❌ Xóa Run")
    if st.button("🗑️ Xóa Run này"):
        try:
            mlflow.delete_run(chosen_run_id)
            st.success(f"✅ Đã xóa run **{chosen_run_name}**! Vui lòng tải lại trang để cập nhật danh sách.")
        except Exception as e:
            st.error(f"❌ Lỗi khi xóa run: {e}")
    
    # Hiển thị thông tin Run chi tiết
    if chosen_run:
        st.subheader(f"📌 Thông tin Run: {chosen_run_name}")
        st.write(f"**Run ID:** {chosen_run_id}")
        st.write(f"**Trạng thái:** {chosen_run.info.status}")
        
        start_time_ms = chosen_run.info.start_time
        start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S") if start_time_ms else "Không rõ"
        st.write(f"**Thời gian chạy:** {start_time}")
        
        # Hiển thị thông số
        if chosen_run.data.params:
            st.write("### ⚙️ Thông số:")
            st.json(chosen_run.data.params)
        if chosen_run.data.metrics:
            st.write("### 📊 Chỉ số:")
            st.json(chosen_run.data.metrics)
        
        # Hiển thị model artifact
        model_path = f"{st.session_state['mlflow_url']}/{chosen_exp.experiment_id}/{chosen_run_id}/artifacts/model"
        st.write("### 📂 Mô hình:")
        st.write(f"📥 [Tải mô hình]({model_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

def upload_data():
    """Chức năng tải lên hoặc tải xuống dữ liệu."""
    st.header("📥 Tải dữ liệu vào hệ thống")
    
    if "data" in st.session_state and st.session_state.data is not None:
        st.warning("🔸 **Dữ liệu đã có sẵn!** Bạn có thể tiếp tục các bước tiếp theo.")
    else:
        source_choice = st.radio("Chọn nguồn dữ liệu:", ["Tải từ OpenML", "Tải lên từ máy"])
        
        if "data" not in st.session_state:
            st.session_state.data = None
        
        if source_choice == "Tải từ OpenML":
            st.markdown("#### 📂 Tải dữ liệu MNIST từ OpenML")
            if st.button("Tải dữ liệu MNIST"):
                with st.status("🔄 Đang tải dữ liệu từ OpenML...", expanded=True) as status:
                    progress = st.progress(0)
                    for percent in range(0, 101, 20):
                        time.sleep(0.5)
                        progress.progress(percent)
                        status.update(label=f"🔄 Đang tải... ({percent}%)")
                    
                    X = np.load("X.npy")
                    y = np.load("y.npy")
                    
                    status.update(label="✅ Tải dữ liệu thành công!", state="complete")
                    st.session_state.data = (X, y)
        
        else:
            st.markdown("#### 📤 Tải ảnh từ thiết bị")
            uploaded_file = st.file_uploader("Chọn ảnh cần tải lên", type=["png", "jpg", "jpeg"])
            
            if uploaded_file is not None:
                with st.status("🔄 Đang kiểm tra và xử lý ảnh...", expanded=True) as status:
                    progress = st.progress(0)
                    for percent in range(0, 101, 25):
                        time.sleep(0.3)
                        progress.progress(percent)
                        status.update(label=f"🔄 Đang xử lý... ({percent}%)")
                    
                    image = Image.open(uploaded_file).convert('L')
                    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)
                    
                    if image.size != (28, 28):
                        status.update(label="❌ Ảnh không đúng kích thước yêu cầu (28x28 px).", state="error")
                    else:
                        status.update(label="✅ Ảnh hợp lệ!", state="complete")
                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                        st.session_state.data = transform(image).unsqueeze(0)
    
    if st.session_state.data is not None:
        st.markdown("#### ✅ Dữ liệu đã sẵn sàng!")
    else:
        st.warning("🔸 Hãy tải dữ liệu trước khi tiếp tục.")
    
    st.markdown("""
    🔹 **Lưu ý:**
    - Chỉ chấp nhận ảnh có kích thước **28x28 pixel (grayscale)**.
    - Khi tải dữ liệu từ OpenML, cần có cột **'label'** để phân loại số từ 0 đến 9.
    - Nếu dữ liệu không đúng định dạng, vui lòng sử dụng tập dữ liệu MNIST.
    """)
def split_data():
    """Chia tập dữ liệu thành Train, Validation và Test."""
    st.markdown("### 📌 Chia dữ liệu Train/Test")
    
    if "data" not in st.session_state or st.session_state.data is None:
        st.error("⚠️ Chưa có dữ liệu! Vui lòng tải dữ liệu trước khi chia tập.")
        return
    
    X, y = st.session_state.data
    total_samples = X.shape[0]
    
    if "data_split_done" not in st.session_state:
        st.session_state.data_split_done = False
    
    num_samples = st.slider("📌 Chọn số lượng mẫu để Train:", 1000, total_samples, min(10000, total_samples))
    test_size = st.slider("📌 Chọn tỷ lệ Test (%):", 10, 50, 20)
    val_size = st.slider("📌 Chọn tỷ lệ Validation (% trong Train):", 0, 50, 15)
    st.write(f"📌 **Tỷ lệ chia:** Test={test_size}%, Validation={val_size}%, Train={100 - test_size - val_size}%")
    
    if st.button("✅ Xác nhận & Lưu") and not st.session_state.data_split_done:
        st.session_state.data_split_done = True
        
        progress = st.progress(0)
        status = st.empty()
        
        status.text("🔄 Đang chọn dữ liệu (0%)")
        
        if num_samples == total_samples:
            X_selected, y_selected = X, y
        else:
            X_selected, _, y_selected, _ = train_test_split(
                X, y, train_size=num_samples, stratify=y if len(np.unique(y)) > 1 else None, random_state=42
            )
        
        progress.progress(25)
        status.text("🔄 Đang chia Train/Test (50%)")
        
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_selected, y_selected, test_size=test_size / 100, stratify=y_selected if len(np.unique(y_selected)) > 1 else None, random_state=42
        )
        
        progress.progress(50)
        status.text("🔄 Đang chia Train/Validation (75%)")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=y_train_full if len(np.unique(y_train_full)) > 1 else None, random_state=42
        )
        
        progress.progress(75)
        
        st.session_state.update({
            "total_samples": num_samples,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "test_size": X_test.shape[0],
            "val_size": X_val.shape[0],
            "train_size": X_train.shape[0]
        })
        
        progress.progress(100)
        status.text("✅ Chia dữ liệu hoàn tất (100%)")
        
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)
    
    elif st.session_state.data_split_done:
        st.info("✅ Dữ liệu đã được chia, không cần chạy lại.")

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_layers, num_nodes, activation):
        super(NeuralNet, self).__init__()
        layers = [nn.Linear(input_size, num_nodes), activation()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_nodes, num_nodes))
            layers.append(activation())
        layers.append(nn.Linear(num_nodes, 10))  # Output layer (10 classes)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train():
    if "mlflow_url" not in st.session_state:
        st.session_state["mlflow_url"] = "https://dagshub.com/Snxtruc/HocMayPython.mlflow"

    mlflow_input()

    if not all(k in st.session_state for k in ["X_train", "X_val", "X_test"]):
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    # Chuyển đổi dữ liệu thành tensor
    X_train = torch.tensor(st.session_state["X_train"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    X_val = torch.tensor(st.session_state["X_val"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32) if st.session_state["X_val"].size > 0 else None
    X_test = torch.tensor(st.session_state["X_test"].reshape(-1, 28 * 28) / 255.0, dtype=torch.float32)
    y_train = torch.tensor(st.session_state["y_train"], dtype=torch.long)
    y_val = torch.tensor(st.session_state["y_val"], dtype=torch.long) if X_val is not None else None
    y_test = torch.tensor(st.session_state["y_test"], dtype=torch.long)

    # Chia batch để huấn luyện thực tế hơn
    batch_size = 64
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Giao diện tùy chỉnh mô hình
    st.header("⚙️ Chọn mô hình & Huấn luyện")
    num_layers = st.slider("Số lớp ẩn", 1, 5, 2)
    num_nodes = st.slider("Số node mỗi lớp", 32, 256, 128)
    activation_func = st.selectbox("Hàm kích hoạt", ["ReLU", "Sigmoid", "Tanh"])
    optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
    learning_rate = st.slider("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f")
    epochs = st.slider("Số epoch", 1, 50, 10)
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state["run_name"] = run_name if run_name else "default_run"

    activation_dict = {"ReLU": nn.ReLU, "Sigmoid": nn.Sigmoid, "Tanh": nn.Tanh}
    activation = activation_dict[activation_func]
    
    model = NeuralNet(28 * 28, num_layers, num_nodes, activation)
    criterion = nn.CrossEntropyLoss()

    optimizer_dict = {
        "Adam": optim.Adam(model.parameters(), lr=learning_rate),
        "SGD": optim.SGD(model.parameters(), lr=learning_rate),
        "RMSprop": optim.RMSprop(model.parameters(), lr=learning_rate),
    }
    optimizer = optimizer_dict[optimizer_choice]

    if st.button("🚀 Huấn luyện mô hình"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_params({
                "num_layers": num_layers,
                "num_nodes": num_nodes,
                "activation": activation_func,
                "optimizer": optimizer_choice,
                "learning_rate": learning_rate,
                "epochs": epochs,
            })

            progress_bar = st.progress(0)
            status_text = st.empty()

            train_losses = []
            val_accuracies = []

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                train_losses.append(epoch_loss / len(train_loader))

                model.eval()
                with torch.no_grad():
                    val_acc = None
                    if X_val is not None:
                        val_preds = model(X_val).argmax(dim=1)
                        val_acc = (val_preds == y_val).float().mean().item()
                        val_accuracies.append(val_acc)

                progress = int((epoch + 1) / epochs * 100)
                progress_bar.progress(progress / 100)

                if val_acc is not None:
                    status_text.text(f"🛠️ Epoch {epoch + 1}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f} | {progress}%")
                else:
                    status_text.text(f"🛠️ Epoch {epoch + 1}/{epochs} | Loss: {train_losses[-1]:.4f} | {progress}%")
                
                time.sleep(0.5)

            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train).argmax(dim=1) == y_train).float().mean().item()
                test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
                val_acc = np.mean(val_accuracies) if val_accuracies else None

            if val_acc is not None:
                st.success(f"📊 **Độ chính xác trên tập validation**: {val_acc:.4f}")
                st.success(f"📊 **Độ chính xác trên tập train**: {train_acc:.4f}")

            mlflow.log_metrics({
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "final_train_loss": train_losses[-1],
            })
            if val_acc is not None:
                mlflow.log_metric("val_accuracy", val_acc)

            # 🔥 Gọi hàm lưu model
            save_model(model, num_layers, num_nodes, activation_func)

            progress_bar.progress(1.0)
            status_text.text("✅ Huấn luyện hoàn tất!")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")

# ✅ HÀM MỚI: Save Model
def save_model(model, num_layers, num_nodes, activation_func):
    model_name = f"{st.session_state['run_name']}_{num_layers}layers_{num_nodes}nodes_{activation_func}"
    
    # Tránh trùng tên model
    if "neural_models" not in st.session_state:
        st.session_state["neural_models"] = []

    existing_names = {m["name"] for m in st.session_state["neural_models"]}
    while model_name in existing_names:
        model_name += "_new"

    model_path = f"{model_name}.pth"
    torch.save({
        "num_layers": num_layers,
        "num_nodes": num_nodes,
        "activation_func": activation_func,
        "model_state_dict": model.state_dict()
    }, model_path)
    
    st.session_state["neural_models"].append({"name": model_name, "model": model_path})
    st.session_state["trained_model"] = model  # ✅ Load ngay vào session_state
    st.success(f"✅ Đã lưu mô hình: `{model_name}`")



# Hàm xử lý ảnh
def preprocess_canvas_image(canvas_result):
    # Kiểm tra nếu canvas trống
    if canvas_result is None or canvas_result.image_data is None:
        return None

    # Chuyển đổi dữ liệu từ canvas thành numpy array
    image_array = np.array(canvas_result.image_data, dtype=np.uint8)

    # Đảm bảo ảnh có đúng 3 kênh (RGB), nếu không, chuyển đổi
    if image_array.shape[-1] == 4:  # Nếu có kênh Alpha (RGBA), loại bỏ
        image_array = image_array[:, :, :3]

    # Chuyển numpy array thành ảnh PIL
    image_pil = Image.fromarray(image_array)

    # Chuyển sang ảnh xám (grayscale) và resize về 28x28
    image_pil = ImageOps.grayscale(image_pil)  
    image_pil = image_pil.resize((28, 28))  

    # Chuyển đổi ảnh thành tensor để đưa vào model
    transform = transforms.Compose([
        transforms.ToTensor(),  
        transforms.Normalize((0.5,), (0.5,))  
    ])
    
    image_tensor = transform(image_pil).view(-1, 28 * 28)  
    return image_tensor


def preprocess_image(canvas_result):
    if canvas_result is None or canvas_result.image_data is None:
        return None
    
    image_array = np.array(canvas_result.image_data, dtype=np.uint8)
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    image_pil = Image.fromarray(image_array)
    image_pil = ImageOps.grayscale(image_pil)  # Chuyển sang grayscale
    image_pil = image_pil.resize((28, 28))  # Resize về 28x28
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_tensor = transform(image_pil).view(-1, 28 * 28)
    return image_tensor

def demo():
    st.title("🔍 Trình nhận diện chữ số viết tay")
    st.write("Vẽ một số bất kỳ từ 0 đến 9, sau đó nhấn **Dự đoán** để xem kết quả!")
    
    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
        st.success("✅ Mô hình đã sẵn sàng để sử dụng!")
    else:
        st.error("⚠️ Không tìm thấy mô hình! Hãy huấn luyện trước.")
        return
    
    if "canvas_key" not in st.session_state:
        st.session_state.canvas_key = str(random.randint(0, 1000000))
    
    if st.button("🔄 Tải lại canvas"):
        st.session_state.canvas_key = str(random.randint(0, 1000000))
    
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.canvas_key,
        update_streamlit=True
    )
    
    if st.button("🚀 Dự đoán"):
        img = preprocess_image(canvas_result)
        if img is not None:
            st.image(
                Image.fromarray((img.numpy().reshape(28, 28) * 255).astype(np.uint8)), 
                caption="Ảnh đã xử lý", 
                width=100
            )
            
            model.eval()
            with torch.no_grad():
                logits = model(img)
                prediction = logits.argmax(dim=1).item()
                confidence_scores = torch.nn.functional.softmax(logits, dim=1)
                max_confidence = confidence_scores.max().item()
            
            col1, col2 = st.columns(2)
            col1.metric("🔢 Số dự đoán", prediction)
            col2.metric("📊 Độ tin cậy", f"{max_confidence:.2%}")
            
            fig, ax = plt.subplots()
            labels = [str(i) for i in range(10)]
            ax.bar(labels, confidence_scores.numpy().flatten(), color="skyblue")
            ax.set_xlabel("Số dự đoán")
            ax.set_ylabel("Mức độ tin cậy")
            ax.set_title("Biểu đồ phân bố xác suất")
            st.pyplot(fig)
        else:
            st.error("⚠️ Vui lòng vẽ một số trước khi bấm Dự đoán!")

def NeuralNetwork():
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
    st.markdown("### 🖊️ MNIST Neural Network App")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Tổng quan",
        "Tải dữ liệu",
        "Chia dữ liệu",
        "Huấn luyện",
        "Thông tin huấn luyện",
        "Demo"
    ])

    with tab2:
        upload_data()
    with tab3:
        split_data()
    with tab4:
        train()
    with tab5:
        show_mlflow_experiments()
    with tab6:
        demo()

def run():
    NeuralNetwork()

if __name__ == "__main__":
    run()
