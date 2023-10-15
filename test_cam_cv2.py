import cv2

# Địa chỉ IP của camera
uri = 'http://192.168.1.108:4747/video'

# Khởi tạo kết nối với camera IP
cap = cv2.VideoCapture(uri)

while True:
    # Đọc frame từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc frame từ camera.")
        break

    # Lật frame theo chiều dọc
    flipped_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Hiển thị frame đã lật
    cv2.imshow('Flipped Frame', flipped_frame)

    # Điều kiện thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
