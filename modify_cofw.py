import numpy as np
import cv2
import h5py
import os

data_root = 'datasets/cofw/'
#mat_file: COFW_train.mat, COFW_test.mat
#img_token: 'IsTr', 'IsT'
#bbox_token: 'bboxesTr', 'bboxesT'
#phis_token: 'phisTr', 'phisT'
def mat_to_files(mat_file, img_token, bbox_token, phis_token, img_dir, gt_txt_file):
	train_mat = h5py.File(mat_file, 'r')
	tr_imgs_obj = train_mat[img_token][:]
	total_num = tr_imgs_obj.shape[1]
	print(total_num)
	eye_occlusion_count = 0
	with open(gt_txt_file, "w+") as trf:
		for i in range(total_num):
			img = train_mat[tr_imgs_obj[0][i]][:]
			bbox = train_mat[bbox_token][:]
			bbox = np.transpose(bbox)[i]
			
			img = np.transpose(img)
			if not os.path.exists(img_dir):
				os.mkdir(img_dir)
			
			gt = train_mat[phis_token][:]
			gt = np.transpose(gt)[i]
			
			content = img_dir + "/{}.jpg,".format(i)
			for k in range(bbox.shape[0]):
				content = content + bbox[k].astype(str) + ' '
			content += ','
			for k in range(gt.shape[0]):
				content = content + gt[k].astype(str) + ' '
			content += '\n'
			if 1.0 in gt[58:75]:
				cv2.imwrite(img_dir + "/{}.jpg".format(i), img)
				trf.write(content)
				eye_occlusion_count += 1
		print(eye_occlusion_count)

mat_to_files(data_root + "COFW_test.mat",
				'IsT', 'bboxesT', 'phisT',
				data_root + "test_eye_occlusion",
				data_root + "test_eye_occlusion/" +"test_ground_true.txt")

# mat_to_files(data_root + "COFW_train.mat",
# 				'IsTr', 'bboxesTr', 'phisTr',
# 				data_root + "train_eye_occlusion",
# 				data_root + "train_eye_occlusion/"+"train_ground_true.txt")