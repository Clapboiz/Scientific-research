Scikit-learn cung cấp một loạt các phương pháp chuẩn hóa dữ liệu thông qua module sklearn.preprocessing. 

```
Dưới đây là một số phương pháp chuẩn hóa dữ liệu phổ biến:
```

```
normalize: Chuẩn hóa dữ liệu bằng cách sử dụng một chuẩn cụ thể, như L1, L2 hoặc Max.
```

```
StandardScaler: Chuẩn hóa dữ liệu bằng cách chuyển dữ liệu thành phân phối chuẩn với trung bình 0 và độ lệch chuẩn 1.
```

```
MinMaxScaler: Đưa dữ liệu về khoảng giá trị cụ thể, thường là [0, 1] hoặc [-1, 1].
```

```
RobustScaler: Chuẩn hóa dữ liệu sử dụng các phân vị và biên độ giúp xử lý nhiễu và dữ liệu ngoại lai.
```

```
PowerTransformer: Áp dụng phép biến đổi mũ để làm cho phân phối dữ liệu gần với phân phối chuẩn.
```

```
QuantileTransformer: Chuyển đổi dữ liệu thành phân phối chuẩn hoặc phân phối thang đo theo phân vị.
```

```
Normalizer: Chuẩn hóa từng mẫu riêng lẻ, thay vì toàn bộ dữ liệu.
```

Mỗi phương pháp này có ứng dụng và tác động khác nhau lên dữ liệu, và lựa chọn phương pháp chuẩn hóa thích hợp phụ thuộc vào bài toán cụ 
thể và tính chất của dữ liệu của bạn.
