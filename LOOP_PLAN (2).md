# LOOP_PLAN.md

## 1. Mục tiêu

Tối ưu website demo **Phân loại Nấm (Ăn được hay Độc)** theo hai yêu cầu:

1. Giao diện đẹp, rõ ràng, dễ đọc và dễ thao tác trên máy tính lẫn điện thoại.
2. Toàn bộ xử lý dự đoán phải nạp và sử dụng đúng cấu hình, bộ tiền xử lý, Encoder và các mô hình đã lưu trong `mo_hinh_nam/`.

Website là sản phẩm học tập. Không trình bày kết quả như lời khuyên an toàn ăn nấm ngoài thực tế.

## 2. Danh sách công việc

| Mã | Công việc | Tiêu chí đạt | Trạng thái |
|---|---|---|---|
| UI01 | Rà soát cấu trúc và phong cách giao diện | Có phân cấp tiêu đề, khoảng cách, màu sắc và bố cục nhất quán; nội dung tiếng Việt dễ đọc | Chờ làm |
| UI02 | Sắp xếp biểu mẫu đặc điểm nấm | Thuộc tính được gom nhóm hợp lý; tên hiển thị bằng tiếng Việt; lựa chọn có ý nghĩa kèm mã dữ liệu | Đang làm — đã có bản giao diện trong `app.py`, cần rà soát |
| UI03 | Hoàn thiện điều khiển Luồng A/B | Nút Luồng A, Luồng B và So sánh cả hai là nút thật, có trạng thái đang chọn và lọc đúng kết quả | Đang làm — đã viết trong `app.py`, cần xác nhận trên bản đang chạy |
| UI04 | Làm rõ trạng thái và kết quả | Có trạng thái nạp mô hình, thông báo lỗi dễ hiểu, nhãn dự đoán và xác suất lớp độc trình bày dễ phân biệt | Chờ làm |
| UI05 | Tối ưu bố cục màn hình nhỏ | Các ô chọn, nút và thẻ kết quả không tràn hoặc chồng lên nhau trên điện thoại | Chờ làm |
| ML01 | Nạp cấu hình và tiền xử lý từ `mo_hinh_nam/` | `app.py` đọc `thong_tin.json` và `tien_xu_ly.pkl`; danh sách cột và giá trị chọn lấy từ cấu hình đã lưu | Đang làm — đã viết trong `app.py`, cần kiểm tra với thư mục mô hình thực tế |
| ML02 | Nạp Encoder và bốn mô hình đã lưu | Nạp `encoder.keras`, `A_KNN.pkl`, `A_GaussianNB.pkl`, `B_KNN.pkl`, `B_GaussianNB.pkl` từ `mo_hinh_nam/` | Đang làm — đã viết trong `app.py`, cần xác nhận tên và đường dẫn tệp |
| ML03 | Giữ đúng quy trình tiền xử lý | Đầu vào → imputer đã lưu → One-Hot Encoder đã lưu; Luồng A dùng One-Hot, Luồng B dùng đầu ra Encoder | Chờ kiểm tra |
| ML04 | Khắc phục tương thích `SimpleImputer` | `transform()` chạy được với bộ imputer đã lưu; không còn lỗi `_fill_dtype`; không thay giá trị imputer bằng thống kê tự tính lại | Đang làm — đã thêm xử lý tương thích trong bản cục bộ, lỗi trên trang hiện tại chưa được xác nhận hết |
| ML05 | Ánh xạ nhãn và xác suất chính xác | `0` hiển thị là “Ăn được”, `1` là “Độc”; xác suất lớp độc lấy theo cột lớp `1` trong `classes_` | Chờ kiểm tra |
| ML06 | Cập nhật kết quả sau khi sửa đầu vào | Mỗi lần bấm Dự đoán dùng các lựa chọn hiện tại; không hiển thị lại kết quả cũ như kết quả mới | Chờ kiểm tra |
| ML07 | Xử lý mẫu minh họa | Nạp mẫu chỉ điền các ô đầu vào; nhãn thật chỉ để tham khảo cho mẫu chưa bị người dùng chỉnh sửa | Chờ kiểm tra |
| ML08 | Báo lỗi thiếu hoặc sai tài nguyên | Nếu thiếu tệp trong `mo_hinh_nam/`, giao diện nêu tên tệp cần có; không hiện traceback dài cho người dùng | Chờ làm |
| ML09 | Đối chiếu logic với notebook Colab | Cùng một hàng đầu vào cho cùng dự đoán ở Colab và website trên bốn cấu hình | Chờ kiểm tra |

## 3. Quy trình thực hiện

1. Đọc mục tiêu, danh sách công việc và nhật ký.
2. Chọn một việc đang Chờ làm hoặc Đang làm. Nếu phát hiện việc còn thiếu trong phạm vi giao diện hoặc logic dùng `mo_hinh_nam`, thêm vào danh sách kèm tiêu chí đạt.
3. Đổi trạng thái thành Đang làm. Tự thực hiện việc và chỉ sửa phần liên quan.
4. Kiểm tra giao diện hoặc chạy dự đoán theo tiêu chí của việc đó.
5. Nếu chưa đạt, ghi lỗi cụ thể, tìm nguyên nhân, sửa rồi kiểm tra lại.
6. Nếu mọi tiêu chí đạt, đổi trạng thái thành Đã xong, ghi nhật ký và quay lại bước 2.

## 4. Điều kiện dừng

- Hết việc trong phạm vi giao diện và logic mô hình thì tổng kết.
- Nếu một việc vẫn lỗi sau 3 lần sửa và kiểm tra lại, đánh dấu Bị kẹt, ghi nguyên nhân rồi chuyển sang việc khác có thể làm độc lập.
- Nếu thiếu tệp trong `mo_hinh_nam/` hoặc thiếu nội dung `thong_tin.json` để xác định tên cột, phiên bản thư viện hay mô hình, ghi rõ tệp/thông tin cần bổ sung.
- Không đánh dấu logic dự đoán đạt nếu vẫn còn lỗi `_fill_dtype` hoặc kết quả website chưa khớp notebook trên cùng đầu vào.

## 5. Nhật ký

| Mã việc | Đã làm gì | Cách kiểm tra và kết quả | Việc tiếp theo |
|---|---|---|---|
| UI02 | Đã tạo bản giao diện tiếng Việt, gom trường theo nhóm và hiển thị nghĩa kèm mã | Có trong bản `app.py` cục bộ; chưa xác nhận trên phiên bản người dùng đang mở | Rà soát bố cục và cách đọc trên điện thoại |
| UI03 | Đã đổi nhãn Luồng A/B thành nút tương tác và thêm nút So sánh cả hai | Có trong bản `app.py` cục bộ; chưa xác nhận đã được cập nhật vào website | Cập nhật ứng dụng đang chạy rồi bấm thử cả ba nút |
| ML01–ML03 | Đã viết phần nạp tài nguyên từ `mo_hinh_nam/` và luồng biến đổi A/B | Chưa đối chiếu trực tiếp với thư mục mô hình đang có trên GitHub | Xác nhận tên, đường dẫn và thứ tự cột từ tài nguyên thực tế |
| ML04 | Đã thêm bước bổ sung `_fill_dtype` cho imputer cũ trong bản cục bộ | Ảnh lỗi mới nhất vẫn cho thấy ứng dụng đang chạy còn lỗi; chưa xác nhận bản sửa đã được dùng | Cập nhật đúng `app.py`, sau đó kiểm tra lại dự đoán |

## 6. Việc tiếp theo

1. Xác nhận thư mục `mo_hinh_nam/` có đúng các tệp mà `app.py` đang nạp.
2. Đưa bản `app.py` mới nhất vào phiên bản website đang chạy.
3. Kiểm tra lỗi `_fill_dtype` đã hết bằng cách bấm Dự đoán.
4. Dùng cùng một mẫu test trong Colab và website, so sánh đầu ra của A-KNN, A-GaussianNB, B-KNN, B-GaussianNB.
5. Sau khi logic đúng, rà soát lần cuối màu sắc, khoảng cách, chữ tiếng Việt và bố cục trên điện thoại.
