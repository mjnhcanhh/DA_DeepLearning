# CHUONG 2. CO SO LY THUYET

## 2.1. Tong quan ve tri tue nhan tao

Tri tue nhan tao la linh vuc nghien cuu va xay dung cac he thong co kha nang thuc hien nhung nhiem vu thuong can den tri thong minh cua con nguoi, chang han nhu nhan dang hinh anh, phan tich du lieu, dua ra du doan, lap ke hoach va ho tro ra quyet dinh. Trong thuc te, AI duoc ung dung rong rai trong y te, giao thong, giao duc, san xuat, an ninh va nhieu linh vuc khac.

Trong do an nay, tri tue nhan tao duoc ung dung vao bai toan phat hien tai nan giao thong tu anh, video va camera mo phong. He thong khong chi dung o viec nhan dien tai nan, ma con mo phong quy trinh ho tro phan ung khan cap sau khi phat hien su co. Cach tiep can nay giup do an co tinh ung dung thuc te cao hon so voi mot chuong trinh chi phan loai anh don gian.

## 2.2. Machine Learning va Deep Learning

Machine Learning la mot nhanh cua tri tue nhan tao, trong do may tinh hoc tu du lieu de tim ra quy luat thay vi phai duoc lap trinh bang cac quy tac co dinh. Mo hinh hoc may se nhan du lieu dau vao, so sanh ket qua du doan voi nhan dung, sau do cap nhat tham so de cai thien kha nang du doan.

Deep Learning la mot nhanh cua Machine Learning, su dung mang no-ron nhieu tang de hoc cac dac trung phuc tap tu du lieu. Trong bai toan xu ly anh, Deep Learning co loi the lon vi co the tu dong hoc cac dac trung nhu duong bien, goc, ket cau, hinh dang va cau truc vat the. Doi voi bai toan phat hien tai nan giao thong, Deep Learning giup mo hinh nhan biet duoc vung co kha nang xay ra tai nan trong khung hinh, ke ca khi boi canh co nhieu phuong tien, vat can va dieu kien anh sang khac nhau.

## 2.3. Mang no-ron nhan tao

Mang no-ron nhan tao gom nhieu lop tinh toan duoc ket noi voi nhau. Mot mang co ban thuong gom lop dau vao, cac lop an va lop dau ra. Moi no-ron nhan du lieu, nhan voi trong so, cong them bias, sau do dua qua ham kich hoat de tao tinh phi tuyen.

Qua trinh huan luyen mang no-ron gom hai buoc chinh. Buoc thu nhat la lan truyen tien, mo hinh dua du lieu dau vao qua cac lop de tao ket qua du doan. Buoc thu hai la lan truyen nguoc, mo hinh tinh sai so thong qua ham mat mat va cap nhat trong so bang thuat toan toi uu. Nho qua trinh lap lai nhieu epoch, mo hinh dan hoc duoc dac trung cua du lieu va cai thien do chinh xac.

## 2.4. Mang no-ron tich chap CNN

CNN la kien truc mang no-ron duoc thiet ke rieng cho du lieu hinh anh. CNN su dung cac bo loc tich chap de trich xuat dac trung khong gian trong anh. Cac lop tich chap o dau mang thuong hoc dac trung don gian nhu canh va goc, trong khi cac lop sau hoc dac trung phuc tap hon nhu hinh dang phuong tien, hien truong va vung va cham.

Mot CNN thuong gom cac thanh phan chinh: convolution layer, activation function, pooling layer, batch normalization, dropout va fully connected layer. Trong cac mo hinh phat hien vat the hien dai, CNN hoac cac bien the cua CNN thuong duoc dung lam backbone de trich xuat dac trung anh truoc khi dua sang phan du doan bounding box va class.

## 2.5. Bai toan phat hien vat the

Phat hien vat the la bai toan xac dinh dong thoi vi tri va loai doi tuong trong anh. Khac voi phan loai anh chi tra ve nhan cho toan bo anh, phat hien vat the can tra ve bounding box, nhan doi tuong va do tin cay cua tung du doan.

Trong do an nay, doi tuong can phat hien la vung tai nan giao thong. Ket qua cua mo hinh gom nhan accident, toa do bounding box va confidence. Neu confidence vuot nguong cau hinh, he thong xem do la mot phat hien hop le va dua ket qua vao module hien thi, benchmark hoac ensemble.

## 2.6. Mo hinh SSD

SSD la mo hinh phat hien vat the mot giai doan. Mo hinh nay du doan truc tiep bounding box va class tu nhieu feature map co do phan giai khac nhau. Nho viec du doan tren nhieu ty le, SSD co kha nang phat hien doi tuong voi kich thuoc khac nhau trong anh.

Uu diem cua SSD la cau truc gon hon cac mo hinh hai giai doan va toc do suy luan tuong doi nhanh, phu hop voi cac he thong can phan hoi gan thoi gian thuc. Trong do an, SSD duoc dung nhu mot trong ba mo hinh phat hien tai nan, ho tro so sanh ket qua va dong gop vao lop Ensemble Decision Layer.

## 2.7. Mo hinh YOLOv12n

YOLO la nhom mo hinh phat hien vat the mot giai doan, co dac diem noi bat la toc do nhanh va kha nang xu ly anh theo thoi gian thuc. Thay vi chia bai toan thanh nhieu buoc rieng biet, YOLO du doan truc tiep bounding box va class tren anh dau vao.

Trong do an, YOLOv12n duoc huan luyen cho 2 lop la accident va normal, su dung image size 640 va batch size 16. Ket qua danh gia cua YOLOv12n dat Precision 0.85752, Recall 0.82116, mAP@0.5 dat 0.87462 va mAP@0.5:0.95 dat 0.54808. Day la mo hinh co ket qua noi bat nhat trong ba mo hinh, dong thoi co kich thuoc checkpoint nho, khoang 20.29 MB, nen phu hop voi bai toan nhan dien tai nan thoi gian thuc.

## 2.8. Mo hinh Faster R-CNN

Faster R-CNN la mo hinh phat hien vat the hai giai doan. Giai doan dau tien, Region Proposal Network de xuat cac vung co kha nang chua doi tuong. Giai doan thu hai, mo hinh phan loai cac vung de xuat va tinh toan lai bounding box chinh xac hon.

Uu diem cua Faster R-CNN la kha nang phat hien on dinh, dac biet voi cac doi tuong can do chinh xac cao. Tuy nhien, do co hai giai doan xu ly nen toc do thuong cham hon cac mo hinh mot giai doan nhu SSD va YOLO. Trong do an, Faster R-CNN su dung backbone ResNet50-FPN, duoc huan luyen voi anh kich thuoc 640 va dat validation mAP@0.5 khoang 0.63775 tai epoch 28.

## 2.9. Ensemble Learning va Weighted Voting

Ensemble Learning la phuong phap ket hop nhieu mo hinh de tao ra ket qua cuoi cung dang tin cay hon. Thay vi phu thuoc vao mot mo hinh don le, he thong lay ket qua tu nhieu mo hinh va tong hop dua tren so phieu, do tin cay va trong so cua tung mo hinh.

Trong do an, module ensemble.py su dung Weighted Voting Ensemble cho 3 mo hinh SSD, YOLOv12n va Faster R-CNN. Ket qua dau vao gom trang thai co phat hien tai nan hay khong, confidence cua tung mo hinh va trang thai load model. He thong tinh diem tong hop theo cong thuc trong so, sau do dua ra quyet dinh cuoi cung. Neu co tu 2 mo hinh dong y, hoac diem weighted vote vuot nguong, hoac Faster R-CNN co confidence rat cao, he thong ket luan co tai nan.

## 2.10. Cac do do danh gia mo hinh

Do an su dung cac chi so danh gia co ban trong bai toan phat hien vat the:

- Precision: ty le du doan tai nan dung tren tong so du doan tai nan.
- Recall: ty le tai nan that duoc mo hinh phat hien dung.
- mAP@0.5: mean Average Precision tai nguong IoU 0.5.
- mAP@0.5:0.95: mAP trung binh tren nhieu nguong IoU tu 0.5 den 0.95.
- FPS: so khung hinh xu ly moi giay.
- Latency: thoi gian xu ly mot anh hoac mot frame.

Doi voi bai toan tai nan giao thong, Precision cao giup giam bao nham, Recall cao giup giam bo sot tai nan. He thong can can bang hai yeu to nay vi bo sot tai nan co the gay hau qua nghiem trong, trong khi bao nham qua nhieu se lam giam do tin cay khi trien khai.

## 2.11. He thong thoi gian thuc va phan ung khan cap

He thong thoi gian thuc la he thong co kha nang xu ly du lieu va dua ra phan hoi trong khoang thoi gian ngan. Trong bai toan giao thong, thoi gian phat hien va phan ung la yeu to quan trong. Khi tai nan xay ra, viec phat hien som co the ho tro dieu phoi luc luong chuc nang nhanh hon, giam thoi gian un tac va tang kha nang ho tro nguoi bi nan.

Do an mo phong mot quy trinh tu phat hien tai nan den tao phieu su co, xac dinh camera, hien thi vi tri tren ban do, tim don vi CSGT va xe cuu thuong gan nhat, sau do cap nhat trang thai xu ly. Day la phan mo rong giup he thong gan voi bai toan van hanh thuc te.

# CHUONG 3. PHAN TICH VA THIET KE HE THONG

## 3.1. Mo ta bai toan

Bai toan cua do an la xay dung he thong AI co kha nang phat hien tai nan giao thong tu anh, video va camera mo phong. Dau vao cua he thong co the la anh upload, video upload, webcam hoac nguon anh demo gan voi camera tren ban do. Dau ra cua he thong la ket qua phat hien tai nan, anh da ve bounding box, confidence, trang thai canh bao va thong tin su co neu tai nan duoc phat hien.

He thong khong chi tap trung vao mo hinh AI ma con xay dung mot ung dung web hoan chinh. Ung dung cho phep nguoi dung chon model, upload du lieu, benchmark 3 mo hinh, quan sat bieu do hieu suat, quet camera tren ban do TP.HCM va quan ly phieu su co.

## 3.2. Yeu cau chuc nang

He thong can dap ung cac chuc nang sau:

- Nhan anh dau vao va phat hien tai nan.
- Nhan video va xu ly tung frame.
- Ho tro webcam de mo phong xu ly truc tiep.
- Cho phep chon thuat toan SSD, YOLOv12n hoac Faster R-CNN.
- Chay benchmark 3 mo hinh tren cung mot anh.
- Hien thi anh ket qua co bounding box va nhan accident.
- Hien thi confidence, FPS, latency va trang thai model.
- Ket hop ket qua 3 mo hinh bang Weighted Voting Ensemble.
- Hien thi ban do camera mo phong tai TP.HCM.
- Quet toan bo camera va phat hien camera co tai nan.
- Tao phieu su co khi phat hien tai nan.
- Mo phong thong bao CSGT va dieu phoi xe cuu thuong.
- Cap nhat trang thai su co: NEW, VERIFIED, DISPATCHED, RESOLVED, FALSE_ALARM.

## 3.3. Yeu cau phi chuc nang

He thong can co giao dien de su dung, phan hoi ro rang va cho phep quan sat ket qua truc quan. Thoi gian xu ly anh can du nhanh de phu hop voi demo thoi gian thuc. Ma nguon duoc tach thanh nhieu module nhu app.py, models.py, detection.py, ensemble.py, camera_map.py va emergency.py de de bao tri.

Ngoai ra, he thong can co kha nang xu ly loi khi model chua load duoc, khi camera chua co nguon demo, khi file upload khong hop le hoac khi ket qua phat hien khong dat nguong confidence.

## 3.4. Kien truc tong the

Kien truc he thong gom 4 lop chinh:

- Lop giao dien: templates/index.html va cac file JavaScript trong static/js.
- Lop backend: Flask app trong app.py, tiep nhan request va dieu phoi xu ly.
- Lop AI inference: models.py load model, detection.py chay nhan dien va ve bounding box.
- Lop ho tro van hanh: ensemble.py tong hop ket qua, camera_map.py quan ly camera, emergency.py quan ly su co va dieu phoi.

Luon xu ly tong quat nhu sau: nguoi dung gui anh/video/camera len giao dien, Flask nhan request, doc frame bang OpenCV, chon model hoac chay ca 3 model, loc ket qua theo confidence va IoU, ve bounding box, tinh ensemble neu can, tra ket qua ve frontend va cap nhat ban do/su co.

## 3.5. Luong xu ly nhan dien tai nan

Luong nhan dien anh gom cac buoc:

1. Nguoi dung upload anh len giao dien.
2. Frontend gui file den route /detect_image hoac /benchmark_image.
3. Backend doc anh bang OpenCV.
4. He thong lay confidence threshold va IoU threshold tu giao dien.
5. Model duoc chon se chay inference.
6. Ket qua du doan duoc loc theo class accident va confidence.
7. He thong ve bounding box len anh.
8. Ket qua duoc tra ve gom anh base64, level, confidence, detection list va thoi gian xu ly.

Neu chay benchmark, he thong se chay lan luot SSD, Faster R-CNN va YOLOv12n tren cung mot anh, sau do tra ve bang so sanh va bieu do.

## 3.6. Luong xu ly phan ung khan cap

Luong phan ung khan cap duoc kich hoat khi module ensemble ket luan co tai nan. He thong xac dinh camera lien quan, lay toa do camera, tao incident trong bo nho, tinh don vi CSGT va xe cuu thuong gan nhat bang cong thuc Haversine, sau do hien thi ke hoach dieu phoi tren giao dien.

Nguoi dung co the xac nhan su co, danh dau da dieu phoi, da xu ly hoac bao nham. Viec nay giup mo phong quy trinh van hanh sau khi he thong AI phat hien tai nan.

## 3.7. Thiet ke cac module chinh

Module app.py la trung tam cua ung dung Flask, chua cac route nhu /detect_image, /benchmark_image, /benchmark_status, /cameras, /scan_cameras, /incidents, /incident/<id>/dispatch, /detect_video va /webcam_frame.

Module models.py khai bao cau hinh 3 thuat toan, duong dan checkpoint, nguong score, class name va cach load model. SSD va YOLOv12n duoc load bang Ultralytics, Faster R-CNN duoc load bang Torchvision.

Module detection.py chiu trach nhiem chay inference, chuan hoa ket qua, ve bounding box, tinh level canh bao va chuyen anh ket qua sang base64 de hien thi tren web.

Module ensemble.py tong hop ket qua tu 3 model bang weighted voting. Trong code hien tai, trong so la SSD 0.25, YOLOv12 0.30 va Faster R-CNN 0.45; nguong ensemble la 0.30, nguong da so la 2 model va nguong confidence cao rieng le la 0.75.

Module camera_map.py quan ly 6 camera mo phong tai TP.HCM, gom CAM_001 den CAM_006, co toa do, quan/huyen va danh sach anh demo.

Module emergency.py quan ly danh sach su co trong bo nho, tinh don vi CSGT va cuu thuong gan nhat, tao dispatch plan va cap nhat trang thai su co.

## 3.8. Thiet ke ban do camera

Ban do camera duoc hien thi bang Leaflet va OpenStreetMap. Cac camera duoc dat tai mot so vi tri trong TP.HCM nhu Nga tu Hang Xanh, Cau Sai Gon, khu vuc HUIT, Nga sau Phu Dong, Cau vuot Cong Hoa va Ham Thu Thiem. Moi camera co id, ten, toa do, quan/huyen, trang thai va danh sach nguon anh/video demo.

Khi nguoi dung chon quet camera, he thong lay ngau nhien mot nguon demo cua moi camera, dua vao module AI va tra ve ket qua. Neu phat hien tai nan, marker camera chuyen sang trang thai canh bao va phieu su co duoc tao.

## 3.9. Thiet ke dieu phoi khan cap

He thong mo phong cac don vi CSGT va xe cuu thuong/bv gan khu vuc TP.HCM. Khi co su co, emergency.py tinh khoang cach tu camera den tung don vi bang cong thuc Haversine. Don vi co khoang cach nho nhat duoc chon vao dispatch plan.

Ke hoach dieu phoi gom ten don vi CSGT, hotline, khoang cach, ETA, don vi cuu thuong, hotline va trang thai dieu phoi. Tren giao dien ban do, nguoi dung co the xem tuyen den hien truong va bam dieu phoi de cap nhat trang thai DISPATCHED.

# CHUONG 4. DU LIEU VA QUA TRINH HUAN LUYEN MO HINH

## 4.1. Gioi thieu bo du lieu

Bo du lieu cua do an gom cac anh giao thong co chua tinh huong tai nan va anh binh thuong. Du lieu duoc luu theo cau truc train, validation va test, kem file nhan dang Pascal VOC XML trong thu muc Train_Fast_CNN. Muc tieu cua bo du lieu la huan luyen mo hinh phat hien vung tai nan trong anh.

Ngoai du lieu huan luyen, he thong con co thu muc static/demo_sources de chua anh demo cho cac camera mo phong. Cac anh nay duoc dung khi quet ban do camera va tao tinh huong phat hien tai nan tren giao dien.

## 4.2. Tien xu ly du lieu

Du lieu anh can duoc kiem tra truoc khi huan luyen. Qua trinh tien xu ly gom kiem tra file anh, kiem tra file nhan, loai bo nhan loi, chuan hoa class, resize anh ve kich thuoc phu hop va chia du lieu thanh cac tap train, validation, test.

Voi bai toan detection, chat luong bounding box anh huong truc tiep den ket qua. Neu nhan qua rong, model co xu huong ve box rong. Neu nhan thieu hoac sai class, model co the bo sot hoac phat hien nham. Do do, buoc lam sach nhan co vai tro quan trong.

## 4.3. Chia tap train, validation va test

Du lieu duoc chia thanh ba phan. Tap train dung de cap nhat trong so cua mo hinh. Tap validation dung de theo doi kha nang hoc trong qua trinh huan luyen va chon checkpoint tot nhat. Tap test dung de danh gia sau cung, khong tham gia vao qua trinh huan luyen.

Trong thu muc du an co script split_dataset_80_10_10.py, cho thay huong chia du lieu theo ty le 80% train, 10% validation va 10% test. Cach chia nay giup mo hinh co du du lieu de hoc, dong thoi van co du lieu doc lap de danh gia.

## 4.4. Huan luyen mo hinh SSD

Mo hinh SSD duoc huan luyen den epoch 32. Checkpoint SSD co ten ssd(1).pth theo thong so duoc cung cap, kich thuoc khoang 182 MB, gom khoang 23.88 trieu tham so va 71 tensor trong state_dict. Optimizer su dung la SGD voi learning rate tai checkpoint 2.5e-05, momentum 0.9 va weight decay 0.0005.

Kien truc SSD suy ra tu checkpoint co backbone dang VGG/SSD, detection head gom classification_head va regression_head. Mo hinh su dung 6 tang feature map de du doan bounding box o nhieu ty le. So anchor tren cac feature map lan luot la 4, 6, 6, 6, 4 va 4. Checkpoint suy ra co 3 class, vi vay can kiem tra lai mapping class khi trien khai de dam bao class accident duoc loc dung.

Hien tai checkpoint SSD chua co san cac chi so mAP, Precision va Recall trong file thong so. Vi vay trong bao cao chi ghi nhan cau hinh va vai tro cua SSD, dong thoi danh dau cac metric nay la chua co danh gia chinh thuc.

## 4.5. Huan luyen mo hinh YOLOv12n

Mo hinh YOLOv12n duoc huan luyen cho bai toan detect voi 2 lop: accident va normal. File checkpoint la yolov12(1).pt, epoch checkpoint la 49 va log huan luyen toi epoch 50. Kich thuoc file khoang 20.29 MB. Mo hinh su dung cau hinh yolov12n.yaml va duoc huan luyen bang Ultralytics version 8.4.41.

Cau hinh huan luyen gom image size 640, batch size 16, optimizer AdamW, learning rate ban dau 0.001, final LR factor 0.01, momentum 0.937, weight decay 0.0005, warmup 3 epoch, workers 8 va device GPU 0. Ket qua danh gia dat Precision 0.85752, Recall 0.82116, mAP@0.5 la 0.87462 va mAP@0.5:0.95 la 0.54808. Cac loss validation cuoi gom val box loss 1.25672, val cls loss 0.72165 va val dfl loss 1.43887.

Ket qua nay cho thay YOLOv12n la mo hinh co hieu nang tot nhat trong 3 mo hinh ve mat chi so mAP va kich thuoc checkpoint. Mo hinh phu hop voi xu ly thoi gian thuc va nen duoc uu tien trong cac tac vu yeu cau toc do.

## 4.6. Huan luyen mo hinh Faster R-CNN

Mo hinh Faster R-CNN su dung backbone ResNet50-FPN/ResNet50 v2, duoc huan luyen voi anh kich thuoc 640. File checkpoint la fast_cnn_model(1).pth. Checkpoint tot nhat duoc luu tai epoch 28, dat best validation mAP@0.5 la 0.6377469233940728. Final test mAP@0.5 theo lan chay truoc khoang 0.6369.

Checkpoint co kich thuoc khoang 329.58 MB, trong khi phan model weights khoang 165 MB. Mo hinh co khoang 43.32 trieu tham so va 404 tensor trong model_state. Cau hinh huan luyen gom batch size 6, learning rate backbone 5e-05, learning rate head 5e-04, momentum 0.9 va weight decay 0.0005.

Lich su huan luyen cho thay train loss giam tu 0.21942 xuong 0.12926, loss classifier giam tu 0.08326 xuong 0.04387, loss box regression giam tu 0.11105 xuong 0.07538, loss objectness giam tu 0.01848 xuong 0.00550 va loss RPN box giam tu 0.00663 xuong 0.00452. mAP@0.5 tang tu 0.52210 len muc tot nhat 0.63775 tai epoch 28. Dieu nay cho thay mo hinh hoc on dinh va co cai thien ro trong qua trinh huan luyen.

## 4.7. Bang so sanh ket qua huan luyen

| Tieu chi | SSD | YOLOv12n | Faster R-CNN |
| --- | --- | --- | --- |
| Loai mo hinh | One-stage detector | One-stage detector | Two-stage detector |
| File checkpoint | ssd(1).pth | yolov12(1).pt | fast_cnn_model(1).pth |
| Epoch | 32 | 49 / log toi 50 | 28 |
| So class | 3 | 2 | 2 |
| Class chinh | accident | accident, normal | background, accident |
| Kich thuoc checkpoint | ~182 MB | ~20.29 MB | ~329.58 MB |
| So tham so | ~23.88M | Chua doc truc tiep | ~43.32M |
| Optimizer | SGD | AdamW | SGD-style |
| Image size | Chua xac dinh trong checkpoint | 640 | 640 |
| mAP@0.5 | Chua co | 0.87462 | 0.63775 validation / 0.6369 test |
| mAP@0.5:0.95 | Chua co | 0.54808 | Chua co |
| Precision | Chua co | 0.85752 | Chua co |
| Recall | Chua co | 0.82116 | Chua co |
| Vai tro | Ho tro detect nhanh va ensemble | Mo hinh nhe, ket qua tot nhat hien tai | Mo hinh doi chieu, on dinh nhung nang hon |

## 4.8. Nhan xet qua trinh huan luyen

Trong ba mo hinh, YOLOv12n cho ket qua noi bat nhat voi mAP@0.5 dat 0.87462, Precision 0.85752 va Recall 0.82116. Mo hinh nay cung co kich thuoc nho nhat, phu hop voi xu ly thoi gian thuc va trien khai demo tren ung dung web.

SSD co uu diem la mo hinh mot giai doan, co kha nang xu ly nhanh va co cau truc phu hop voi bai toan detection. Tuy nhien, checkpoint hien tai chua co metric danh gia chinh thuc, nen can chay danh gia lai tren tap test de bo sung mAP, Precision va Recall.

Faster R-CNN co ket qua on dinh, train loss giam deu va mAP tang trong qua trinh huan luyen. Tuy nhien, mo hinh co kich thuoc lon va toc do cham hon, phu hop hon vai tro doi chieu va tang do tin cay trong ensemble.

# CHUONG 5. XAY DUNG UNG DUNG

## 5.1. Cong nghe su dung

He thong duoc xay dung bang Python va Flask cho backend. Cac mo hinh AI duoc trien khai bang PyTorch, Torchvision va Ultralytics. OpenCV duoc dung de doc anh, video, webcam va xu ly frame. NumPy ho tro xu ly du lieu so. Phan giao dien duoc xay dung bang HTML, CSS va JavaScript.

He thong ban do su dung Leaflet va OpenStreetMap de hien thi camera mo phong. Chart.js duoc dung de ve bieu do benchmark, FPS, latency va stress test. Du lieu giua frontend va backend duoc trao doi chu yeu bang JSON va anh base64.

## 5.2. Cau truc thu muc du an

Cau truc chinh cua du an gom:

- app.py: file Flask chinh, chua route va dieu phoi xu ly.
- models.py: khai bao cau hinh va load cac mo hinh AI.
- detection.py: chay inference, loc ket qua va ve bounding box.
- ensemble.py: ket hop ket qua 3 mo hinh bang weighted voting.
- camera_map.py: quan ly camera mo phong va nguon du lieu demo.
- emergency.py: quan ly phieu su co va dieu phoi khan cap.
- templates/index.html: giao dien web chinh.
- static/js/main.js: xu ly upload anh, video, webcam va hien thi ket qua.
- static/js/benchmark.js: xu ly benchmark 3 mo hinh.
- static/js/charts.js: hien thi bieu do hieu suat va stress test.
- static/js/map.js: hien thi ban do camera va tuyen dieu phoi.
- static/js/emergency.js: hien thi va cap nhat phieu su co.
- Models: chua cac checkpoint da huan luyen.
- Train_Fast_CNN: chua du lieu va script huan luyen Faster R-CNN.

## 5.3. Module load mo hinh AI

Module models.py chiu trach nhiem khai bao ALGO_CONFIG cho SSD, YOLOv12n va Faster R-CNN. Moi model co ten hien thi, danh sach checkpoint ung vien, framework, mau hien thi, nguong score, so bounding box toi da va class name.

SSD va YOLOv12n duoc load bang Ultralytics. Faster R-CNN duoc load bang Torchvision va co co che tu nhan dien so class tu checkpoint. Module nay cung luu trang thai model da load, duong dan model va loi load neu co. Nho do, giao dien benchmark co the hien thi ro model nao dang san sang va model nao bi loi.

## 5.4. Module nhan dien tai nan

Module detection.py thuc hien cac tac vu nhan dien. Dau vao la frame anh dang NumPy array. Module chay model, loc ket qua theo confidence threshold, loc class accident, gioi han so detection va ve bounding box len anh.

Ket qua tra ve gom danh sach detection, confidence cao nhat, trang thai level va anh ket qua. Trong giao dien, level 2 duoc xem la tai nan, level 1 la canh bao yeu va level 0 la binh thuong. De giam nhieu, he thong cho phep dieu chinh confidence va IoU threshold tu giao dien.

## 5.5. Module Ensemble Decision Layer

Module ensemble.py tong hop ket qua tu 3 mo hinh. Moi mo hinh dong gop mot phieu neu phat hien tai nan va co confidence hop le. Diem weighted score duoc tinh bang tong confidence nhan voi trong so, sau do chuan hoa theo tong trong so cua cac model da load.

Logic ket luan co tai nan khi thoa mot trong cac dieu kien: co it nhat 2 model dong y, weighted score vuot nguong 0.30, hoac Faster R-CNN co confidence cao hon 0.75. Ket qua ensemble gom accident, level, ensemble_score, votes, reason, thong tin tung model va weights_used.

## 5.6. Module ban do camera TP.HCM

Module camera_map.py quan ly 6 camera mo phong tai TP.HCM. Moi camera co id, ten, toa do, quan/huyen va danh sach file anh/video demo. Cac camera duoc hien thi tren ban do Leaflet. Khi quet camera, he thong lay ngau nhien anh/video tu thu muc static/demo_sources va dua vao pipeline nhan dien.

Neu camera khong co nguon hop le, he thong tra ve trang thai no_source. Neu co tai nan, marker camera doi mau va popup hien thi thong tin canh bao. Ket qua nay cung duoc dung de tao phieu su co trong module emergency.

## 5.7. Module quan ly su co khan cap

Module emergency.py luu danh sach incident trong bo nho. Moi incident gom ma su co, thoi gian, camera, toa do, trang thai, diem ensemble, so phieu model, ly do ket luan, thong tin tung model, anh bang chung, nguon du lieu va ke hoach dieu phoi.

Trang thai su co gom NEW, VERIFIED, DISPATCHED, RESOLVED va FALSE_ALARM. Nguoi dung co the cap nhat trang thai tren giao dien. Cach thiet ke nay giup mo phong vong doi xu ly mot su co tu luc phat hien den luc ket thuc.

## 5.8. Module dieu phoi CSGT va xe cuu thuong

He thong khai bao danh sach don vi CSGT va don vi cuu thuong/bv tai TP.HCM. Khi co tai nan, module emergency.py tinh don vi gan nhat dua tren toa do camera. Khoang cach duoc tinh bang cong thuc Haversine, sau do uoc luong thoi gian den hien truong voi toc do do thi trung binh va thoi gian chuan bi.

Ke hoach dieu phoi tra ve gom ten don vi, hotline, quan/huyen, khoang cach va ETA. Tren ban do, map.js co the ve tuyen dieu phoi tu don vi den camera tai nan. Neu goi dispatch, incident duoc cap nhat trang thai DISPATCHED va ghi them dispatch_log.

## 5.9. Giao dien he thong

Giao dien he thong gom nhieu khu vuc chuc nang. Khu vuc Live Detection cho phep upload anh, video hoac webcam. Khu vuc Benchmark cho phep chay 3 mo hinh tren cung mot anh va hien thi bang so sanh. Khu vuc Hieu suat hien thi bieu do FPS, latency va stress test. Khu vuc Ban do giam sat hien thi camera TP.HCM va ket qua quet camera. Khu vuc Su co hien thi danh sach incident, dispatch plan va cac nut cap nhat trang thai.

Giao dien duoc thiet ke de nguoi dung co the quan sat ro ket qua AI, so sanh mo hinh va theo doi quy trinh phan ung khan cap trong cung mot ung dung.

# CHUONG 6. THU NGHIEM VA DANH GIA

## 6.1. Kich ban thu nghiem

Do an co the duoc thu nghiem theo cac kich ban sau:

- Upload mot anh co tai nan de kiem tra kha nang phat hien.
- Upload mot anh binh thuong de kiem tra bao nham.
- Upload nhieu anh de kiem tra xu ly hang loat.
- Upload video de kiem tra xu ly tung frame.
- Chay webcam de mo phong thoi gian thuc.
- Benchmark SSD, YOLOv12n va Faster R-CNN tren cung mot anh.
- Chay stress test nhieu lan de do FPS va latency trung binh.
- Quet toan bo camera tren ban do.
- Tao incident khi phat hien tai nan.
- Dieu phoi CSGT va xe cuu thuong, sau do cap nhat trang thai su co.

## 6.2. Thu nghiem nhan dien anh

Khi upload anh, he thong doc anh va chay model dang duoc chon. Neu phat hien tai nan, anh ket qua se co bounding box bao quanh vung tai nan, kem nhan accident va confidence. Nguoi dung co the thay doi confidence threshold va IoU threshold de quan sat su thay doi cua ket qua.

Ket qua thu nghiem cho thay he thong phat hien tot cac truong hop tai nan ro rang. Voi cac anh co hien truong phuc tap, nhieu xe chong len nhau hoac goc chup xau, bounding box co the rong hon vung tai nan thuc te. Dieu nay lien quan den chat luong nhan va dac diem phuc tap cua du lieu giao thong.

## 6.3. Thu nghiem video va webcam

Voi video, backend doc tung frame va gui ket qua ve frontend theo dang stream. Voi webcam, giao dien goi route /webcam_frame de lay frame va cap nhat ket qua lien tuc. Cac mo hinh mot giai doan nhu YOLOv12n va SSD phu hop hon voi xu ly video do toc do nhanh. Faster R-CNN co do on dinh tot nhung toc do cham hon va phu hop hon voi vai tro doi chieu.

De cai thien ket qua video, he thong co the ap dung them xu ly theo chuoi frame. Trong static/js/main.js da co co che lam muot ket qua theo frame gan nhau, giup giam truong hop mot frame bi bo sot nhung cac frame lan can deu phat hien tai nan.

## 6.4. Thu nghiem benchmark 3 mo hinh

Chuc nang benchmark chay SSD, Faster R-CNN va YOLOv12n tren cung mot anh. Ket qua hien thi gom anh dau ra cua tung model, confidence, thoi gian xu ly, FPS va level canh bao. Giao dien cung ve bieu do de so sanh toc do va do tin cay giua cac model.

Bang benchmark giup nguoi dung thay ro su khac biet giua mo hinh mot giai doan va hai giai doan. YOLOv12n co loi the ve kich thuoc va metric danh gia. Faster R-CNN co vai tro doi chieu va on dinh. SSD ho tro nhanh nhung can bo sung metric chinh thuc.

## 6.5. Thu nghiem ban do camera

Khi vao tab ban do, he thong hien thi 6 camera mo phong tai TP.HCM. Khi bam quet camera, moi camera duoc gan mot nguon anh/video demo va chay qua pipeline ensemble. Ket qua tra ve cho biet camera nao binh thuong, camera nao chua co anh demo va camera nao phat hien tai nan.

Neu co tai nan, marker tren ban do doi sang trang thai canh bao, popup hien thi ket qua va phieu su co duoc tao. Cach thu nghiem nay mo phong tinh huong he thong giam sat nhieu camera giao thong cung luc.

## 6.6. Thu nghiem phan ung khan cap

Khi incident duoc tao, giao dien hien thi ma su co, thoi gian, camera, toa do, diem ensemble, so phieu model, anh bang chung va ke hoach dieu phoi. Nguoi dung co the xac nhan su co, dieu phoi CSGT va xe cuu thuong, danh dau da xu ly hoac bao nham.

Ket qua thu nghiem cho thay module emergency mo phong duoc quy trinh tu phat hien tai nan den dieu phoi luc luong. Mac du chua ket noi API that cua CSGT, tong dai 115 hay camera giao thong that, chuc nang nay the hien duoc y tuong ung dung AI vao quy trinh ho tro van hanh.

## 6.7. So sanh cac mo hinh

| Tieu chi | SSD | YOLOv12n | Faster R-CNN |
| --- | --- | --- | --- |
| Nhom mo hinh | One-stage | One-stage | Two-stage |
| Toc do ly thuyet | Nhanh | Nhanh | Cham hon |
| Kich thuoc checkpoint | ~182 MB | ~20.29 MB | ~329.58 MB |
| mAP@0.5 | Chua co | 0.87462 | 0.63775 validation |
| Precision | Chua co | 0.85752 | Chua co |
| Recall | Chua co | 0.82116 | Chua co |
| Uu diem | Ho tro detect nhanh | Nhe, metric tot nhat | On dinh, doi chieu tot |
| Han che | Chua co metric chinh thuc | Can test them tren du lieu thuc | Nang va cham hon |

Nhan xet tong hop: YOLOv12n la mo hinh noi bat nhat hien tai do co mAP@0.5 cao va checkpoint nho. Faster R-CNN phu hop de tang do tin cay va lam model doi chieu. SSD giu vai tro bo sung trong ensemble va can duoc danh gia lai tren tap test de co so lieu day du.

## 6.8. Danh gia he thong

Uu diem cua he thong:

- Co ung dung web hoan chinh, khong chi la code train model.
- Ho tro anh, video, webcam va camera mo phong.
- Tich hop 3 mo hinh phat hien tai nan.
- Co chuc nang benchmark va bieu do hieu suat.
- Co ban do camera TP.HCM.
- Co module ensemble de tang do tin cay.
- Co module incident va dispatch plan de mo phong phan ung khan cap.

Han che cua he thong:

- Camera trong ban do la camera mo phong, chua ket noi camera that.
- Canh bao CSGT va xe cuu thuong moi la mo phong, chua gui API/SMS/email that.
- SSD va Faster R-CNN chua co day du Precision, Recall va mAP@0.5:0.95.
- Du lieu huan luyen phu thuoc vao tap du lieu co san.
- Mot so bounding box co the rong hoac chua sat vung tai nan.
- Incident dang luu trong bo nho, chua co database lau dai.

# CHUONG 7. KET LUAN VA HUONG PHAT TRIEN

## 7.1. Ket qua dat duoc

Do an da xay dung duoc he thong AI nhan dien tai nan giao thong va ho tro phan ung khan cap theo thoi gian thuc. He thong co the nhan dien tai nan tu anh, video, webcam va camera mo phong. Ba mo hinh SSD, YOLOv12n va Faster R-CNN duoc tich hop vao ung dung, dong thoi co co che Weighted Voting Ensemble de tong hop ket qua.

Ve mat ung dung, he thong da xay dung duoc giao dien web bang Flask, ho tro upload du lieu, benchmark model, hien thi bieu do, quan ly ban do camera TP.HCM, tao phieu su co va mo phong dieu phoi CSGT/xe cuu thuong. Day la mot quy trinh tuong doi day du, bao gom ca phan AI va phan xu ly sau khi phat hien.

Ve mat mo hinh, YOLOv12n dat ket qua tot nhat voi mAP@0.5 la 0.87462, Precision 0.85752 va Recall 0.82116. Faster R-CNN dat validation mAP@0.5 khoang 0.63775 va test mAP@0.5 khoang 0.6369. SSD duoc huan luyen den epoch 32 voi khoang 23.88 trieu tham so, dong vai tro ho tro trong ensemble va can danh gia them metric chinh thuc.

## 7.2. Han che

Mac du he thong da dat duoc muc tieu cua do an, van con mot so han che. Thu nhat, du lieu huan luyen chua bao quat het cac tinh huong giao thong ngoai thuc te nhu mua, dem, camera mo, goc quay xa hoac tai nan bi che khu. Thu hai, mot so metric cua SSD va Faster R-CNN chua day du, can chay danh gia chuan hoa tren cung tap test.

Thu ba, he thong chua ket noi camera giao thong that, chua co API ban do giao thong thoi gian thuc va chua gui canh bao that den co quan chuc nang. Thu tu, incident hien dang duoc luu tam trong bo nho, nen khi restart server du lieu se mat. Thu nam, Faster R-CNN co kich thuoc lon va toc do cham hon, can toi uu neu muon chay tren may cau hinh thap.

## 7.3. Huong phat trien

Trong tuong lai, he thong co the phat trien theo cac huong sau:

- Ket noi camera IP hoac camera giao thong that.
- Bo sung database de luu lich su su co lau dai.
- Tich hop gui canh bao qua email, SMS, Telegram hoac API tong dai.
- Tich hop ban do giao thong thoi gian thuc va du lieu GPS.
- Danh gia lai tat ca mo hinh tren cung tap test de co metric dong nhat.
- Dieu chinh trong so ensemble theo ket qua thuc nghiem, co the tang trong so YOLOv12n vi mo hinh nay co mAP cao nhat.
- Cai thien bo du lieu, lam sach bounding box va bo sung anh trong nhieu dieu kien anh sang/thoi tiet.
- Toi uu model de chay tren thiet bi bien nhu Jetson Nano, Raspberry Pi hoac server mini.
- Them chuc nang tracking de theo doi su co qua nhieu frame lien tiep.
- Them chuc nang phan loai muc do nghiem trong cua tai nan.
- Trien khai he thong tren cloud hoac server noi bo de nhieu nguoi dung co the truy cap.

## 7.4. Ket luan chung

Do an cho thay kha nang ung dung Deep Learning vao bai toan phat hien tai nan giao thong va ho tro phan ung khan cap. Bang viec ket hop cac mo hinh SSD, YOLOv12n va Faster R-CNN, he thong co the so sanh nhieu cach tiep can va dua ra quyet dinh tong hop bang ensemble. Phan ban do camera va phieu su co giup do an gan hon voi mot he thong giam sat giao thong thong minh trong thuc te.

Neu duoc tiep tuc phat trien voi du lieu lon hon, ket noi camera that va co che canh bao that, he thong co the tro thanh mot cong cu ho tro huu ich cho viec giam sat giao thong, rut ngan thoi gian phat hien tai nan va nang cao hieu qua phan ung khan cap.
