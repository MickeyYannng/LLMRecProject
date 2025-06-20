from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 绝对路径
file_path = r'D:\美琪的魔仙堡\代码文件\LLMRecProject\data\ml-100k\u.data'

reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)

print("RMSE:", accuracy.rmse(predictions))
