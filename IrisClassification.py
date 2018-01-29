from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import MulticlassMetrics
import decimal

#Dataset Input
input_data = sc.textFile("/FileStore/tables/mzx9zg9f1479616815582/iris.csv")

#Dataset Parse Function
def parse(line):
  field = line.split(",")
  sep_length = float(field[0])
  sep_width = float(field[1])
  pet_length = float(field[2])
  pet_width = float(field[3])
  tar_class = field[4]
  if tar_class == 'Iris-versicolor':
    tarFinal = 0
  else:
    tarFinal = 1
  return LabeledPoint(tarFinal,(sep_length,sep_width,pet_length,pet_width)
  
#Dataset Parsing
newData = input_data.filter(lambda x : not x.startswith('sepallength'))
parsed_data = newData.map(parse)

#TrainTest Split
training, test = parsed_data.randomSplit([0.7, 0.3])

#Model Fit
model = LogisticRegressionWithLBFGS.train(training)

#Model Evaluation
predAndLabel = test.map(lambda x: (float(model.predict(x.features)), x.label))
metrics = MulticlassMetrics(predAndLabel)
true_pos=1.0 * predAndLabel.map(lambda x : (x[0],x[1])).filter(lambda x: x[0] == 1.0).filter(lambda x : x[1] == 1.0).count()
false_pos=1.0 * predAndLabel.map(lambda x : (x[0],x[1])).filter(lambda x: x[0] == 1.0).filter(lambda x : x[1] == 0.0).count()
false_neg = 1.0 * predAndLabel.map(lambda x : (x[0],x[1])).filter(lambda x: x[0] == 0.0).filter(lambda x : x[1] == 1.0).count()
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)accuracy = 1.0 * predAndLabel.filter(lambda (x, v): x == v).count() / test.count()
percentage = "%"

#Result
print('The accuracy of the model is {}{}'.format(100*round(accuracy,2),percentage))
print("The model's precision is {}{}").format(100*round(precision,2),percentage)
print("The model's recall is {}{}").format(100*round(recall,2),percentage)
