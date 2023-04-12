# CNN là gì?
CNN là từ viết tắt của cụm Convolutional Neural Network hay là mạng nơ ron tích chập. 

Đây là mô hình vô cùng tiên tiến được áp dụng nhiều trong lĩnh vực học sâu Deep learning. 

Mạng CNN cho phép người dùng xây dựng những hệ thống phân loại và dự đoán với độ chính xác cực cao. 

Hiện nay,mạng CNN được ứng dụng nhiều hơn trong xử lý ảnh, cụ thể là nhận diện đối tượng trong ảnh.

Mỗi hidden layer được gọi là fully connected layer, tên gọi theo đúng ý nghĩa, mỗi node trong hidden
layer được kết nối với tất cả các node trong layer trước. Cả mô hình được gọi là fully connected
neural network (FCN).

![image](https://user-images.githubusercontent.com/112185647/231429916-ca4ab3fe-02c8-4d7d-93ee-411bbf40f611.png)

## Quy tắc, convolutional layer?
Stride là khoảng cách giữa các vùng tương tự trong một ma trận đầu vào khi thực hiện phép tích chập (convolutional neural network) hoặc phép trượt cửa sổ (sliding window) trên đầu vào đó. Nếu Stride bằng 1, các vùng tương tự sẽ trùng lắp nhau. Nếu Stride lớn hơn 1, các vùng sẽ không trùng lắp và có khoảng cách giữa chúng.

Padding là thêm các giá trị 0 vào các biên của ma trận đầu vào trước khi thực hiện phép tích chập hoặc phép trượt cửa sổ. Mục đích của việc này là để đảm bảo rằng kích thước của đầu ra sẽ bằng kích thước của đầu vào. Nếu không có Padding, đầu ra sẽ có kích thước nhỏ hơn đầu vào.

Với mỗi kernel khác nhau ta sẽ học được những đặc trưng khác nhau của ảnh, nên trong mỗi
convolutional layer ta sẽ dùng nhiều kernel để học được nhiều thuộc tính của ảnh. Vì mỗi kernel
cho ra output là 1 matrix nên k kernel sẽ cho ra k output matrix. Ta kết hợp k output matrix này lại thành 1 tensor 3 chiều có chiều sâu k

        * Hiểu đơn giản kernel là 1 ma trận m*n con di chuyển lần lượt qua ma trận lớn (Image gốc)

Output của convolutional layer đầu tiên sẽ thành input của convolutional layer tiếp theo.

![image](https://user-images.githubusercontent.com/112185647/231528406-bc9036df-7541-4d8b-8ab0-9712dad51587.png)

        * Lưu ý: Output của convolutional layer sẽ qua hàm non-linear activation function trước khi trở thành input của convolutional layer tiếp theo. (Khi train đến đó thì sẽ có hàm lọc vd như: relu, softmax,... tùy thuộc vào bài toán thì ta sẽ chọn hàm lọc tương ứng)

## Pooling layer
Pooling layer thường được dùng giữa các convolutional layer, để giảm kích thước dữ liệu nhưng vẫn giữ được các thuộc tính quan trọng. Việc giảm kích thước dữ liệu giúp giảm các phép tính toán trong model

        * Lưu ý: Thường thì pooling sẽ dùng size(2,2), stride = 2, padding = 0, Khi đó output width và height của dữ liệu giảm đi một nửa, depth thì được giữ nguyên. 
                 Pooling giúp tăng hiệu suất train.
                 Các biến trong pooling có thể được tính từ: Max, avg.

Sau khi ảnh được truyền qua nhiều convolutional layer và pooling layer thì model đã học được tương đối các đặc điểm của ảnh (ví dụ mắt, mũi, khung mặt,...) thì tensor của output của layer cuối cùng, kích thước H*W*D, sẽ được chuyển về 1 vector kích thước (H*W*D, 1)

![image](https://user-images.githubusercontent.com/112185647/231531168-88e54a00-9363-4d35-b8ac-d2153d630455.png)

Sau đó ta dùng các fully connected layer để kết hợp các đặc điểm của ảnh để ra được output của model.

         * 1 số pre-trained models là VGG 16, Resnet, Inception, ....
