# CNN là gì?
CNN là từ viết tắt của cụm Convolutional Neural Network hay là mạng nơ ron tích chập. 

Đây là mô hình vô cùng tiên tiến được áp dụng nhiều trong lĩnh vực học sâu Deep learning. 

Mạng CNN cho phép người dùng xây dựng những hệ thống phân loại và dự đoán với độ chính xác cực cao. 

Hiện nay,mạng CNN được ứng dụng nhiều hơn trong xử lý ảnh, cụ thể là nhận diện đối tượng trong ảnh.

Mỗi hidden layer được gọi là fully connected layer, tên gọi theo đúng ý nghĩa, mỗi node trong hidden
layer được kết nối với tất cả các node trong layer trước. Cả mô hình được gọi là fully connected
neural network (FCN).

Về cơ bản thiết kế của một mạng nơ ron tích chập 2 chiều có dạng như sau:

INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC

Các kí hiệu []N, []M hoặc []*K ám chỉ các khối bên trong [] có thể lặp lại nhiều lần liên tiếp nhau. M, K là số lần lặp lại. Kí hiệu -> đại diện cho các tầng liền kề nhau mà tầng đứng trước sẽ làm đầu vào cho tầng đứng sau. Dấu ? sau POOL để thể hiện tầng POOL có thể có hoặc không sau các khối tích chập.

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

Lưu ý:

    Cả tensorflow và keras đều có BatchNormalization, điểm khác nhau giữa 2 cái này là:
          + tensorflow thì nhiều tham số truyền vào hơn keras.
          + Keras hỗ trợ cho việc đồng bộ hóa BatchNormalization giữa các thiết bị (devices) khác nhau, còn tensorflow thì không.
          + keras.layers.BatchNormalization()     #keras
          + tf.keras.layers.BatchNormalization()  #tensorflow
          + Import 2 thư viện này:
                 - from tensorflow.keras.layers import BatchNormalization
                 - from keras.layers import BatchNormalization
    Lớp BatchNormalization thường được đặt sau một lớp Convolutional hoặc Dense và trước một hàm kích hoạt (activation function) trong mô hình
    Khi đã thực hiện chuẩn hóa bằng cách /255 trước đó thì có thể không cần BatchNormalization nữa

## Gan
GAN là viết tắt “generative adversarial network”, hướng tới việc sinh ra dữ liệu mới sau quá trình học. GAN có thể tự sinh ra một khuôn mặt mới, một con người, một đoạn văn, chữ viết, bản nhạc giao hưởng hay những thứ tương tự thế. Thế làm cách nào để GAN học và làm được điều đó, chúng ta cần phải điểm qua một vài khái niệm.

Các mô hình Machine Learning có thể được phân chia thành lớp mô hình phân biệt (Discriminative) và mô hình sinh (Generative). Đây chỉ là một cách phân chia trong vô số các cách phân chia khác như: mô hình học có giám sát (supervised learning)/học không giám sát (unsupervised learning), mô hình tham số (parametric)/mô hình phi tham số (non parametric), mô hình đồ thị (graphic)/mô hình phi đồ thị (non-graphic),….

![image](https://user-images.githubusercontent.com/112185647/231714743-b17ef216-a349-4d69-aaeb-a435c606edef.png)

Generator: Học cách sinh ra dữ liệu giả để lừa mô hình Discriminator. Để có thể đánh lừa được Discriminator thì đòi hỏi mô hình sinh ra output phải thực sự tốt. Do đó chất lượng ảnh phải càng như thật càng tốt.

Discriminator: Học cách phân biệt giữa dữ liệu giả được sinh từ mô hình Generator với dữ liệu thật. Discriminator như một giáo viên chấm điểm cho Generator biết cách nó sinh dữ liệu đã đủ tinh xảo để qua mặt được Discriminator chưa và nếu chưa thì Generator cần tiếp tục phải học để tạo ra ảnh thật hơn. Đồng thời Discriminator cũng phải cải thiện khả năng phân biệt của mình vì chất lượng ảnh được tạo ra từ Generator càng ngày càng giống thật hơn. Thông qua quá trình huấn luyện thì cả Generator và Discriminator cùng cải thiện được khả năng của mình.

![image](https://user-images.githubusercontent.com/112185647/231713133-21131383-1d69-4e2e-b1f2-6a2635e71229.png)

Generator và Discriminator tương tự như hai người chơi trong bài toán zero-sum game trong lý thuyết trò chơi. Ở trò chơi này thì hai người chơi xung đột lợi ích. Hay nói cách khác, thiệt hại của người này chính là lợi ích của người kia. Mô hình Generator tạo ra dữ liệu giả tốt hơn sẽ làm cho Discriminator phân biệt khó hơn và khi Discriminator phân biệt tốt hơn thì Generator cần phải tạo ra ảnh giống thật hơn để qua mặt Discriminator. Trong zero-sum game, người chơi sẽ có chiến lược riêng của mình, đối với Generator thì đó là sinh ra ảnh giống thật và Discriminator là phân loại ảnh thật/giả. Sau các bước ra quyết định của mỗi người chơi thì zero-sum game sẽ đạt được cân bằng Nash tại điểm cân bằng (Equilibrium Point).
### Generator
![image](https://user-images.githubusercontent.com/112185647/231719150-0fdc00c5-f581-4e4a-819d-ec9ad1d5c60c.png)

Generator về bản chất là một mô hình sinh nhận đầu vào là một tập hợp các véc tơ nhiễu được khởi tạo ngẫu nhiên theo phân phối Gaussian. Ở một số lớp mô hình GAN tiên tiến hơn, input có thể làm một dữ liệu chẳng hạn như bức ảnh, đoạn văn bản hoặc đoạn âm thanh. Nhưng ở đây với mục đích làm quen và tìm hiểu GAN đầu vào được giả sử là véc tơ nhiễu như trong bài báo gốc Generative Adversarial Nets của tác giả Ian J.Goodfellow.

Từ tập véc tơ đầu vào ngẫu nhiên, mô hình generator là một mạng học sâu có tác dụng biến đổi ra bức ảnh giả ở output. Bức ảnh giả này sẽ được sử dụng làm đầu vào cho kiến trúc Discriminator.
### Discriminator
![image](https://user-images.githubusercontent.com/112185647/231719982-4474e52a-7fb4-40e2-b502-833d9094dc0c.png)

Mô hình Discriminator sẽ có tác dụng phân biệt ảnh input là thật hay giả. Nhãn của mô hình sẽ là thật nếu ảnh đầu vào của Discriminator được lấy tập mẫu huấn luyện và giả nếu được lấy từ output của mô hình Generator. Về bản chất đây là một bài toán phân loại nhị phân (binary classification) thông thường. Để tính phân phối xác suất cho output cho Discriminator chúng ta sử dụng hàm sigmoid.

## Imagedatagenerator 
Hiện nay trong deep learning thì vấn đề dữ liệu có vai trò rất quan trọng. Chính vì vậy có những lĩnh vực có ít dữ liệu để cho việc train model thì rất khó để tạo ra được kết quả tốt trong việc dự đoán. Do đó người ta cần đến một kỹ thuật gọi là tăng cường dữ liệu (data augmentation) để phục vụ cho việc nếu bạn có ít dữ liệu, thì bạn vẫn có thể tạo ra được nhiều dữ liệu hơn dựa trên những dữ liệu bạn đã có. Ví dụ như hình dưới, đó là các hình được tạo ra thêm từ một ảnh gốc ban đầu.

       + Original (Ảnh gốc): dĩ nhiên rồi, bao giờ mình cũng có ảnh gốc
       
       + Flip (Lật): lật theo chiều dọc, ngang miễn sao ý nghĩa của ảnh (label) được giữ nguyên hoặc suy ra được. Ví dụ nhận dạng quả bóng tròn, thì mình lật kiểu gì cũng ra quả bóng. Còn với nhận dạng chữ viết tay, lật số 8 vẫn là 8, nhưng 6 sẽ thành 9 (theo chiều ngang) và không ra số gì theo chiều dọc. Còn nhận dạng ảnh y tế thì việc bị lật trên xuống dưới là không bao giờ sảy ra ở ảnh thực tế --> không nên lật làm gì
       
       + Random crop (Cắt ngẫu nhiên): cắt ngẫu nhiên một phần của bức ảnh. Lưu ý là khi cắt phải giữ thành phần chính của bức ảnh mà ta quan tâm. Như ở nhận diện vật thể, nếu ảnh được cắt không có vật thể, vậy giá trị nhãn là không chính xác.
       
       + Color shift (Chuyển đổi màu): Chuyển đổi màu của bức ảnh bằng cách thêm giá trị vào 3 kênh màu RGB. Việc này liên quan tới ảnh chụp đôi khi bị nhiễu --> màu bị ảnh hưởng.
       
       + Noise addition (Thêm nhiễu): Thêm nhiễu vào bức ảnh. Nhiễu thì có nhiều loại như nhiễu ngẫu nhiên, nhiễu có mẫu, nhiễu cộng, nhiễu nhân, nhiễu do nén ảnh, nhiễu mờ do chụp không lấy nét, nhiễu mờ do chuyển động có thể kể hết cả ngày.
       
       + Information loss (Mất thông tin): Một phần của bức hình bị mất. Có thể minh họa trường hợp bị che khuất.
       
       + Constrast change (Đổi độ tương phản): thay độ tương phản của bức hình, độ bão hòa
       
       + Geometry based: Đủ các thể loại xoay, lật, scale, padding, bóp hình, biến dạng hình,
       
       + Color based: giống như trên, chi tiết hơn chia làm (i) tăng độ sắc nét, (ii) tăng độ sáng, (iii) tăng độ tương phản hay (iv) đổi sang ảnh negative - âm bản.
       
       + Noise/occlusion: Chi tiết hơn các loại nhiễu, như mình kể trên còn nhiều lắm. kể hết rụng răng.
       
       + Whether: thêm tác dụng cảu thời tiết như mưa, tuyết, sương mờ,

![image](https://user-images.githubusercontent.com/112185647/231939667-8644e337-4ebb-4857-ab57-d97f99768b0e.png)

     * Lưu ý: Cả Gan và Imagedatagenerator thì nếu dữ liệu đủ lớn thì có thể không cần dùng 2 phương pháp này, còn dữ liệu ít thì nên dùng để tăng hiệu suất cho mô hình.
              Việc sử dụng data aumgentation nên thực hiện ngẫu nhiên trong quá trình huấn luyện. Và việc dùng GAN là tạo dữ liệu ko có trước, có thể có tác dụng phụ.
              
