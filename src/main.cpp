#include <iostream>
#include <Eigen/Dense>
#include "rapidcsv.h"

using namespace Eigen;
using namespace std;

int main() {
    try {
        // Đọc dữ liệu từ tệp CSV sử dụng RapidCSV, với dấu phân tách là dấu chấm phẩy
        cout << "Đang thử mở tệp CSV..." << endl;
        rapidcsv::Document doc("../wine_quality_dataset/winequality-red.csv", rapidcsv::LabelParams(0, -1), rapidcsv::SeparatorParams(';'));
        cout << "Đã mở tệp CSV thành công." << endl;

        // Lấy số dòng và số cột trong tập dữ liệu
        size_t rowCount = doc.GetRowCount();
        size_t colCount = doc.GetColumnCount();

        cout << "Số lượng dòng: " << rowCount << endl;
        cout << "Số lượng cột: " << colCount << endl;

        // Lấy dữ liệu của các cột để chuyển sang ma trận Eigen
        MatrixXd data(rowCount, colCount);
        for (size_t i = 0; i < colCount; ++i) {
            vector<string> col = doc.GetColumn<string>(i);
            for (size_t j = 0; j < rowCount; ++j) {
                try {
                    // Chuyển đổi từ chuỗi sang số
                    data(j, i) = stod(col[j]);
                } catch (const std::invalid_argument &e) {
                    cerr << "Lỗi khi chuyển đổi giá trị: " << col[j] << " tại dòng " << j + 1 << " và cột " << i + 1 << endl;
                    data(j, i) = 0.0; // Gán giá trị 0.0 cho các giá trị không hợp lệ
                } catch (const std::out_of_range &e) {
                    cerr << "Giá trị vượt quá phạm vi: " << col[j] << " tại dòng " << j + 1 << " và cột " << i + 1 << endl;
                    data(j, i) = 0.0; // Gán giá trị 0.0 cho các giá trị không hợp lệ
                }
            }
        }

        // In ra dữ liệu đầu tiên để kiểm tra
        cout << "Dữ liệu ban đầu (5 dòng đầu tiên):" << endl;
        cout << data.topRows(5) << endl << endl;

        // Chuẩn hóa dữ liệu (trừ đi giá trị trung bình và chia cho độ lệch chuẩn)
        VectorXd mean = data.colwise().mean();
        MatrixXd centered = data.rowwise() - mean.transpose();
        VectorXd std_dev = ((centered.array().square().colwise().sum()) / (data.rows() - 1)).sqrt();
        MatrixXd normalized = centered.array().rowwise() / std_dev.transpose().array();

        // In ra dữ liệu sau khi chuẩn hóa (5 dòng đầu tiên)
        cout << "Dữ liệu sau khi chuẩn hóa (5 dòng đầu tiên):" << endl;
        cout << normalized.topRows(5) << endl;

    } catch (const std::exception &e) {
        cerr << "Đã xảy ra lỗi: " << e.what() << endl;
    }

    return 0;
}
