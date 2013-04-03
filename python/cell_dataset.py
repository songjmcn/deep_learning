import dataset.create_dataset as dataset
train_dir='d:/data/train'
train_label='d:/data/ICPR2012_Cells_Classification_Contest/cell_label_information_Train.txt'
test_dir='d:/data/test'
test_label='d:/data/ICPR2012_Cells_Classification_Contest/cell_label_information_Test.txt'
dataset_path='d:/data/cell_dataset.gz'
dataset.create_dataset(train_dir, train_label, test_dir, test_label, dataset_path)