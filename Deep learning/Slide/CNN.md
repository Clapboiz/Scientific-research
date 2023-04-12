## CNN là gì?
CNN là từ viết tắt của cụm Convolutional Neural Network hay là mạng nơ ron tích chập. 

Đây là mô hình vô cùng tiên tiến được áp dụng nhiều trong lĩnh vực học sâu Deep learning. 

Mạng CNN cho phép người dùng xây dựng những hệ thống phân loại và dự đoán với độ chính xác cực cao. 

Hiện nay,mạng CNN được ứng dụng nhiều hơn trong xử lý ảnh, cụ thể là nhận diện đối tượng trong ảnh.

Mỗi hidden layer được gọi là fully connected layer, tên gọi theo đúng ý nghĩa, mỗi node trong hidden
layer được kết nối với tất cả các node trong layer trước. Cả mô hình được gọi là fully connected
neural network (FCN).

![image](https://user-images.githubusercontent.com/112185647/231429916-ca4ab3fe-02c8-4d7d-93ee-411bbf40f611.png)

## Quy tắc?
Stride là khoảng cách giữa các vùng tương tự trong một ma trận đầu vào khi thực hiện phép tích chập (convolutional neural network) hoặc phép trượt cửa sổ (sliding window) trên đầu vào đó. Nếu Stride bằng 1, các vùng tương tự sẽ trùng lắp nhau. Nếu Stride lớn hơn 1, các vùng sẽ không trùng lắp và có khoảng cách giữa chúng.
Padding là thêm các giá trị 0 vào các biên của ma trận đầu vào trước khi thực hiện phép tích chập hoặc phép trượt cửa sổ. Mục đích của việc này là để đảm bảo rằng kích thước của đầu ra sẽ bằng kích thước của đầu vào. Nếu không có Padding, đầu ra sẽ có kích thước nhỏ hơn đầu vào.
