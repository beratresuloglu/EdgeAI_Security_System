cmake_minimum_required(VERSION 3.10)
project(Berat_EdgeAI_Final)

set(CMAKE_CXX_STANDARD 17)

# OpenCV'nin tam yolunu manuel tanımlıyoruz (Ninja hata vermesin diye)
set(OPENCV_BASE "C:/opencv/build")
set(OPENCV_INCLUDE_DIRS "${OPENCV_BASE}/include")
set(OPENCV_LIBS "${OPENCV_BASE}/x64/vc16/lib/opencv_world4120d.lib") # Sürüm numaranı kontrol et (4100, 480, vb.)

include_directories(${OPENCV_INCLUDE_DIRS})

add_executable(SecuritySystem main.cpp)

# Doğrudan .lib dosyasını bağlıyoruz
target_link_libraries(SecuritySystem ${OPENCV_LIBS})