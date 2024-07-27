import cv2
import numpy as np

def region_of_interest(img, vertices):                              # Vertice hinh dang cua da giac 
    mask = np.zeros_like(img)                                       # Pixel khong quan tam co gia tri 0
    cv2.fillPoly(mask, vertices, 255)                               # Ve da giac mau trang (255) len mask
    masked = cv2.bitwise_and(img, mask)                             # Ap dung len anh goc chi giu lai phan anh nam trong vung quan tam (vung mau trang) 
    return masked                                                   # Tra ve anh da duoc ap dung mat na

def draw_lines(img, lines, color=[255, 0, 0], thickness=7):         # Dinh nghia duong thang voi 4 thong so (anh,duong can ve,mau do,do day)
    for line in lines:
        for x1,y1,x2,y2 in line:                                    # Toa do diem dau toi diem cuoi
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)     # Ve doan thang tu diem dau toi diem cuoi dua theo mau sac va do day

def process_image(image):
    # Chuyển đổi ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                  # Chuyen doi anh sang grayscale de de dang xu li hinh anh
    
    # Xác định ROI
    height = image.shape[0]                                                                          
    width = image.shape[1]
    roi_vertices = [(0, height), (width/2, height/2), (width, height)]      # Vung quan tam duoc tao tu 2 duong line tao thanh mot tam giac (3 dinh tao thanh) 
                                                                            # Goc toa do nam o goc tren ben trai        

    # Áp dụng Canny edge detection
    edges = cv2.Canny(gray, 50, 150)                                        # Xac dinh canh tu bien grey voi duoi nguong 50 se bo qua va tren nguong 150 se duoc coi la canh                                  
    
    # Áp dụng mask ROI
    masked_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))                    #Tạo một mặt nạ dựa trên các đỉnh roi_vertices.
                                                                                                    # Áp dụng mặt nạ này lên ảnh edges để chỉ giữ lại các cạnh nằm trong vùng ROI.
                                                                                                    # Kết quả là hình ảnh masked_edges, chứa các cạnh nằm trong vùng quan tâm được xác định.    
    
    # Áp dụng Hough Transform
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)      # Vẽ các đoạn thẳng với các tham số cụ thể để xác định độ phân giải, ngưỡng, chiều dài tối thiểu, khoảng cách tối đa giữa các đoạn thẳng
    
    # Tạo một ảnh trống để vẽ các đường
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)                      # Tạo ra ảnh trống cùng kích thước ảnh gốc để vẽ các đường thẳng đã được phát hiện với các thông số hàng, cột, kênh 
    
    # Vẽ các đường phát hiện được
    if lines is not None:
        draw_lines(line_image, lines)
    
    # Kết hợp ảnh gốc với ảnh chứa các đường    
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)                                          # Kết quả là sự pha trộn giữa ảnh gốc và ảnh chứa các đường       
    
    return result

# Đọc ảnh đầu vào
image = cv2.imread(r'C:\Users\admin\Downloads\Lane.png')

# Xử lý ảnh
result = process_image(image)

# Hiển thị kết quả
cv2.imshow('Lane Detection', result)
cv2.waitKey(0)
cv2.destroyAllWindows()