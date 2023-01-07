# Scientific-research
<h1>#1 ML</h1>
<h2>Source: https://users.soict.hust.edu.vn/khoattq/ml-dm-course/ </h2>
<h3>
        Library:
</h3>
<h5>    Numpy</h5>
<h5>    Matplotlib</h5>
<h5>    Pandas</h5>

<h4>Học có giám sát và học không giám sát</h4>
+ Học có giám sát là bài toán hồi quy
<h4>Học có giám sát là bài toán phân loại</h4>
Bao gồm: Multiclass (Phân loại nhiều lớp) <br>
                + có rất nhiều nhãn nhưng 1 y chỉ ứng với 1 x <br>
                + check Spam filtering, y in {spam, normal} <br>
                + Discovery of network attacks <br>
                + Financial risk estimation: y in{high, normal, no} <br>
        Multilabel (Phân loại đa nhãn)
                        + ex: Birds nest tree <br>
                + Output y is subset of labels (Mỗi output là 1 tập nhỏ các lớp, mỗi quan sát x có thể có nhiều nhãn) <br>
                +Image tagging y = {bird, nest, tree} <br>
                +sentiment analysis <br>
                        + Prediction of stock indices <br>
 <h4>Học không giám sát</h4>
        + Clustering data into cluster (vân tay , ...) <br>
        + Comunity detection <br>
        + Trend detection <br> 

Overfitting: làm rất tốt với tập dữ liệu train nhưng với tập thực tế thì lại rất tệ (quá khít, quá khớp) <br>
Underfitting: Làm tệ với cả 2 tập dữ liệu  <br>
        + Regularization là 1 mô hình giúp chúng ta huấn luyện được mô hình tốt và tránh được hiện tượng overfitting <br>

<h4>Tiền xử lý dữ liệu</h4>
        Thời gian dành cho phân tích dữ liệu
                + Thu thập dữ liệu: 19%
                + Thu xếp và làm sạch dữ liệu: 60%
                + Tạo tập dữ liệu huấn luyện: 3%
                + Khai phá: 9%
                + Cải thiện thuật toán: 4%
                + Khác: 5%
        Tiền xử lý dữ liêu để làm gì? Thuận tiện trong việc lưu trữ, truy vấn
        Các mô hình học máy thường làm  việc với các mô hình dữ liệu có cấu trúc ma trận vector, chuỗi,....
        
