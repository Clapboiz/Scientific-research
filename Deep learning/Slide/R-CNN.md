# R-CNN
Mạng nơ-ron tích chập theo vùng, hay các vùng với đặc trưng CNN (R-CNN) là một hướng tiếp cận tiên phong ứng dụng mô hình sâu cho bài toán phát hiện vật thể

Đầu tiên, các mô hình R-CNN sẽ chọn một số vùng đề xuất từ ảnh (ví dụ, các khung neo cũng là một phương pháp lựa chọn) và sau đó gán nhãn hạng mục và khung chứa (ví dụ, các giá trị độ dời) cho các vùng này.

Tiếp đến, các mô hình này sử dụng CNN để thực hiện lượt truyền xuôi nhằm trích xuất đặc trưng từ từng vùng đề xuất. Sau đó, ta sử dụng các đặc trưng của từng vùng được đề xuất để dự đoán hạng mục và khung chứa. Dưới đây mô tả một mô hình R-CNN.

![image](https://user-images.githubusercontent.com/112185647/232552237-995363c2-2960-40b1-844b-d3dde38d3a82.png)

Cụ thể, R-CNN có bốn phần chính sau:

    Tìm kiếm chọn lọc trên ảnh đầu vào để lựa chọn các vùng đề xuất tiềm năng. Các vùng đề xuất thông thường sẽ có nhiều tỷ lệ với hình dạng và kích thước khác nhau. Hạng mục và khung chứa nhãn gốc sẽ được gán cho từng vùng đề xuất.
    
    Sử dụng một mạng CNN đã qua tiền huấn luyện, ở dạng rút gọn, đặt trước tầng đầu ra. Mạng này biến đổi từng vùng đề xuất thành các đầu vào có chiều phù hợp với mạng và thực hiện các lượt truyền xuôi để trích xuất đặc trưng từ các vùng đề xuất tương ứng.

    Các đặc trưng và nhãn hạng mục của từng vùng đề xuất được kết hợp thành một mẫu để huấn luyện các máy vector hỗ trợ cho phép phân loại vật thể. Ở đây, mỗi máy vector hỗ trợ được sử dụng để xác định một mẫu có thuộc về một hạng mục nào đó hay không.
    
    Các đặc trưng và khung chứa được gán nhãn của mỗi vùng đề xuất được kết hợp thành một mẫu để huấn luyện mô hình hồi quy tuyến tính, để phục vụ dự đoán khung chứa nhãn gốc.
   
Mặc dù các mô hình R-CNN sử dụng các mạng CNN đã được tiền huấn luyện để trích xuất các đặc trưng ảnh một cách hiệu quả, điểm hạn chế chính yếu đó là tốc độ chậm. Có thể hình dung, với hàng ngàn vùng đề xuất từ một ảnh, ta cần tới hàng ngàn phép tính truyền xuôi từ mạng CNN để phát hiện vật thể. Khối lượng tính toán nặng nề khiến các mô hình R-CNN không được sử dụng rộng rãi trong các ứng dụng thực tế.

## Fast R-CNN
Điểm nghẽn cổ chai chính về hiệu năng của R-CNN đó là việc trích xuất đặc trưng cho từng vùng đề xuất một cách độc lập. Do các vùng đề xuất này có độ chồng lặp cao, việc trích xuất đặc trưng một cách độc lập sẽ dẫn đến một số lượng lớn các phép tính lặp lại. Fast R-CNN cải thiện R-CNN bằng cách chỉ thực hiện lượt truyền xuôi qua mạng CNN trên toàn bộ ảnh.
              
     * "Điểm nghẽn cổ chai" (bottle-neck) là thuật ngữ được sử dụng để chỉ một điểm trong một quy trình hoặc hệ thống mà nếu có vấn đề xảy ra tại điểm đó, thì sẽ gây ra sự cố cho toàn bộ hệ thống.
              
![image](https://user-images.githubusercontent.com/112185647/232557087-422c5606-bbd0-44d5-819f-92f0dcea9636.png)

Mô tả mạng Fast R-CNN. Các bước tính toán chính yếu được mô tả như sau:

So với mạng R-CNN, mạng Fast R-CNN sử dụng toàn bộ ảnh làm đầu vào cho CNN để trích xuất đặc trưng thay vì từng vùng đề xuất. Hơn nữa, mạng này được huấn luyện như bình thường để cập nhật tham số mô hình. Do đầu vào là toàn bộ ảnh, đầu ra của mạng CNN có kích thước  1×c×h1×w1.

Giả sử thuật toán tìm kiếm chọn lọc chọn ra n vùng đề xuất, kích thước khác nhau của các vùng này chỉ ra rằng vùng quan tâm (regions of interests - RoI) tại đầu ra của CNN có kích thước khác nhau. Các đặc trưng có cùng kích thước phải được trích xuất từ các vùng quan tâm này (giả sử có chiều cao là h2 và chiều rộng là  w2). Mạng Fast R-CNN đề xuất phép gộp RoI (RoI pooling), nhận đầu ra từ CNN và các vùng quan tâm làm đầu vào rồi ghép nối các đặc trưng được trích xuất từ mỗi vùng quan tâm làm đầu ra có kích thước  n×c×h2×w2.

Tầng kết nối đầy đủ được sử dụng để biến đổi kích thước đầu ra thành  n×d , trong đó  d được xác định khi thiết kế mô hình.

Khi dự đoán hạng mục, kích thước đầu ra của tầng kết nối đầy đủ lại được biến đổi thành  n×q và áp dụng phép hồi quy softmax ( q là số lượng hạng mục). Khi dự đoán khung chứa, kích thước đầu ra của tầng đầy đủ lại được biến đổi thành  n×4 . Nghĩa là ta dự đoán hạng mục và khung chứa cho từng vùng đề xuất.

## Faster R-CNN
Để có kết quả phát hiện đối tượng chính xác, Fast R-CNN thường đòi hỏi tạo ra nhiều vùng đề xuất khi tìm kiếm chọn lọc. Faster R-CNN thay thế tìm kiếm chọn lọc bằng mạng đề xuất vùng. Mạng này làm giảm số vùng đề xuất, trong khi vẫn đảm bảo phát hiện chính xác đối tượng.

![image](https://user-images.githubusercontent.com/112185647/232567560-2b4abf8a-2f07-4cde-b498-4ba9764b728e.png)

minh họa mô hình Faster R-CNN. So với Fast R-CNN, Faster R-CNN chỉ thay thế phương pháp sản sinh các vùng đề xuất từ tìm kiếm chọn lọc sang mạng đề xuất vùng. Những phần còn lại trong mô hình không đổi. Quy trình tính toán của mạng đề xuất vùng được mô tả chi tiết dưới đây:

Dùng một tầng tích chập  3×3 với đệm bằng 1 để biến đổi đầu ra của CNN và đặt số kênh đầu ra bằng  c. Bằng cách này, mỗi phần tử trong ánh xạ đặc trưng mà CNN trích xuất ra từ bức ảnh là một đặc trưng mới có độ dài bằng  c.

Lấy mỗi phần tử trong ánh xạ đặc trưng làm tâm để tạo ra nhiều khung neo có kích thước và tỷ lệ khác nhau, sau đó gán nhãn cho chúng.

Lấy những đặc trưng của các phần tử có độ dài c ở tâm khung neo để phân loại nhị phân (là vật thể hay là nền) và dự đoán khung chứa tương ứng cho các khung neo.

Sau đó, sử dụng triệt phi cực đại (non-maximum suppression) để loại bỏ các khung chứa có kết quả giống nhau của hạng mục “vật thể”. Cuối cùng, ta xuất ra các khung chứa dự đoán là các vùng đề xuất rồi đưa vào tầng gộp RoI.

Lưu ý rằng, vì là một phần của mô hình Faster R-CNN, nên mạng đề xuất vùng được huấn luyện cùng với phần còn lại trong mô hình. Ngoài ra, trong đối tượng Faster R-CNN còn chứa các hàm dự đoán hạng mục và khung chứa trong bài toán phát hiện vật thể, cũng như các hàm dự đoán hạng mục nhị phân và khung chứa cho các khung neo trong mạng đề xuất vùng. Sau cùng, mạng đề xuất vùng có thể học được cách sinh ra những vùng đề xuất có chất lượng cao, giảm đi số lượng vùng đề xuất trong khi vẫn giữ được độ chính xác khi phát hiện vật thể.

## Mask R-CNN
Nếu dữ liệu huấn luyện được gán nhãn với các vị trí ở cấp độ từng điểm ảnh trong bức hình, thì mô hình Mask R-CNN có thể sử dụng hiệu quả các nhãn chi tiết này để cải thiện độ chính xác của việc phát hiện đối tượng.

![image](https://user-images.githubusercontent.com/112185647/232568156-3284a33a-c184-4f21-a993-0edaec1eda1e.png)

Có thể thấy Mask R-CNN là một sự hiệu chỉnh của Faster R-CNN. Mask R-CNN thay thế tầng tổng hợp RoI bằng tầng căn chỉnh RoI (RoI alignment layer). Điều này cho phép sử dụng phép nội suy song tuyến tính (bilinear interpolation) để giữ lại thông tin không gian trong ánh xạ đặc trưng, làm cho Mask R-CNN trở nên phù hợp hơn với các dự đoán cấp điểm ảnh. 

Lớp căn chỉnh RoI xuất ra các ánh xạ đặc trưng có cùng kích thước cho mọi RoI. Điều này không những dự đoán các lớp và khung chứa của RoI, mà còn cho phép chúng ta bổ sung một mạng nơ-ron tích chập đầy đủ (fully convolutional network) để dự đoán vị trí cấp điểm ảnh của các đối tượng. Chúng tôi sẽ mô tả cách sử dụng mạng nơ-ron tích chập đầy đủ để dự đoán ngữ nghĩa cấp điểm ảnh ở phần sau của chương này.

# Tổng quan 
Mô hình R-CNN chọn ra nhiều vùng đề xuất và sử dụng CNN để thực hiện tính toán truyền xuôi rồi trích xuất đặc trưng từ mỗi vùng đề xuất. Sau đó dùng các đặc trưng này để dự đoán hạng mục và khung chứa của những vùng đề xuất.

Fast R-CNN cải thiện R-CNN bằng cách chỉ thực hiện tính toán truyền xuôi CNN trên toàn bộ bức ảnh. Mạng này sử dụng một tầng gộp RoI để trích xuất các đặc trưng có cùng kích thước từ các vùng quan tâm có kích thước khác nhau.

Faster R-CNN thay thế tìm kiếm chọn lọc trong Fast R-CNN bằng mạng đề xuất vùng. Điều này làm giảm số lượng vùng đề xuất tạo ra, nhưng vẫn đảm bảo độ chính xác khi phát hiện đối tượng.

Mask R-CNN có cấu trúc cơ bản giống Faster R-CNN, nhưng có thêm một mạng nơ-ron tích chập đầy đủ giúp định vị đối tượng ở cấp điểm ảnh và cải thiện hơn nữa độ chính xác của việc phát hiện đối tượng.

|Kiến trúc |Tốc độ | Độ chính xác |	Ưu điểm |	Nhược điểm|
|-------|-------|-------|-------|-------|
| CNN|	N/A| 	Thấp|- Dễ dàng triển khai.<br>- Hoạt động tốt với ảnh có kích thước nhỏ.<br>- Thời gian huấn luyện ngắn.<br>- Tương đối đơn giản.|- Độ chính xác thấp.<br>- Không xử lý được các vị trí và đối tượng khác nhau trong ảnh.<br>- Không có cơ chế để tìm ra các đối tượng trong ảnh.|
|R-CNN|Chậm|Cao|- Tốt cho việc phát hiện đối tượng với độ chính xác cao.<br>- Tự động tìm ra các khu vực ứng viên có đối tượng trong ảnh.<br>- Hiệu suất được cải thiện bằng cách sử dụng các đặc trưng được học sâu từ mạng CNN.|- Tốc độ chậm do phải sử dụng mạng CNN để rút trích các đặc trưng của từng khu vực ứng viên.<br>- Không hiệu quả với ảnh có kích thước lớn vì phải áp dụng mạng CNN nhiều lần.<br>- Yêu cầu nhiều tài nguyên tính toán và bộ nhớ do phải xử lý các khu vực ứng viên một cách riêng lẻ.|
|Fast R-CNN | Nhanh| Cao|- Tốc độ nhanh hơn so với R-CNN vì chỉ cần chạy một lần mạng CNN cho toàn bộ ảnh.<br>- Tự động tìm ra các khu vực ứng viên có đối tượng trong ảnh.<br>- Hiệu suất được cải thiện bằng cách sử dụng các đặc trưng được học sâu từ mạng CNN.|- Yêu cầu nhiều tài nguyên tính toán và bộ nhớ do phải xử lý các khu vực ứng viên một cách riêng lẻ.<br>- Không hiệu quả với ảnh có kích thước lớn vì phải áp dụng mạng CNN nhiều lần.|
|Faster R-CNN|Tương đối nhanh|Cao|- Giải quyết vấn đề tốc độ của R-CNN và Fast R-CNN bằng cách sử dụng một mạng CNN duy nhất để tìm ra các khu vực ứng viên và các đặc trưng của chúng.<br>- Tự động tìm ra các khu vực ứng viên có đối tượng trong ảnh.<br>- Độ chính xác cao hơn so với R-CNN và Fast R-CNN.|- Yêu cầu nhiều tài nguyên tính toán và bộ nhớ do phải xử lý các khu vực ứng viên một cách riêng lẻ.<br>- Không hiệu quả với ảnh có kích thước lớn vì phải áp dụng mạng CNN nhiều lần.<br>- Khó khăn trong việc huấn luyện do phải tối ưu hóa hai phần riêng biệt: mô hình tìm kiếm khu vực ứng viên và mô hình phân loại đối tượng.|
|Mask R-CNN|Chậm|Cao|- Đồng thời giải quyết vấn đề phát hiện đối tượng và phân đoạn ảnh.<br>- Tự động tìm ra các khu vực ứng viên có đối tượng trong ảnh.<br>- Tăng cường độ chính xác của phân đoạn ảnh bằng cách kết hợp thông tin về đối tượng với các đặc trưng của ảnh.<br>- Độ chính xác cao hơn so với các mô hình chỉ phát hiện đối tượng.|- Tốc độ chậm do phải sử dụng mạng CNN để rút trích các đặc trưng của từng khu vực ứng viên.<br>- Yêu cầu nhiều tài nguyên tính toán và bộ nhớ do phải xử lý các khu vực ứng viên một cách riêng lẻ.<br>- Khó khăn trong việc huấn luyện do phải tối ưu hóa ba phần riêng biệt: mô hình tìm kiếm khu vực ứng viên|

Dưới đây là sắp xếp các thuật toán theo mức độ mạnh từ trên xuống dưới:

Mask R-CNN: Thuật toán mạnh nhất trong các thuật toán đã nêu, cho phép nhận diện vật thể và định vị đối tượng, đồng thời cũng cho phép phân đoạn ảnh (segmentation).

Faster R-CNN: Cho phép nhận diện và định vị đối tượng với độ chính xác cao hơn so với các thuật toán trước đó, và cải thiện tốc độ xử lý bằng cách sử dụng mô hình RPN.

R-CNN: Đây là thuật toán đầu tiên trong series R-CNN, cho phép nhận diện và định vị đối tượng, sử dụng phương pháp lấy mẫu (selective search) để đưa ra các vùng ảnh chứa đối tượng.

Fast R-CNN: Cải tiến từ R-CNN, Fast R-CNN cải thiện tốc độ xử lý bằng cách sử dụng một mạng nơ-ron tích chập (CNN) để trích xuất đặc trưng của ảnh, và áp dụng phương pháp RoI pooling để đưa ra các vùng ảnh chứa đối tượng.

CNN: Là thuật toán cơ bản nhất trong series này, chỉ đơn giản là một mạng nơ-ron tích chập (CNN), được sử dụng để trích xuất đặc trưng của ảnh. Các thuật toán sau đó đều cải thiện và phát triển trên cơ sở của CNN.

Lưu ý rằng sự xếp hạng này chỉ mang tính chất đánh giá chung, và phụ thuộc vào mục đích sử dụng cũng như bài toán cụ thể. Các thuật toán này có ưu điểm và hạn chế riêng, và sẽ phù hợp với các bài toán và mục đích khác nhau.
