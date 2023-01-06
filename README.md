# Scientific-research
<h1>#1 ML</h1>
<h3>
        Library:
</h3>
<h5>    Numpy</h5>
<h5>    Matplotlib</h5>
<h5>    Pandas</h5>

Học có giám sát và học không giám sát
+ Học có giám sát là bài toán hồi quy
<h4>Học có giám sát là bài toán phân loại</h4>
Bao gồm: Multiclass (Phân loại nhiều lớp)
                + có rất nhiều nhãn nhưng 1 y chỉ ứng với 1 x
                + check Spam filtering, y in {spam, normal}
                + Discovery of network attacks
                + Financial risk estimation: y in{high, normal, no}
        Multilabel (Phân loại đa nhãn)
                        + ex: Birds nest tree
                + Output y is subset of labels (Mỗi output là 1 tập nhỏ các lớp, mỗi quan sát x có thể có nhiều nhãn)
                +Image tagging y = {bird, nest, tree}
                +sentiment analysis
                        + Prediction of stock indices
 <h4>Học không giám sát</h4>
        + Clustering data into cluster (vân tay , ...)
        + Comunity detection
        + Trend detection

Overfitting: làm rất tốt với tập dữ liệu train nhưng với tập thực tế thì lại rất tệ (quá khít, quá khớp)
Underfitting: Làm tệ với cả 2 tập dữ liệu 
        + Regularization là 1 mô hình giúp chúng ta huấn luyện được mô hình tốt và tránh được hiện tượng overfitting
        
