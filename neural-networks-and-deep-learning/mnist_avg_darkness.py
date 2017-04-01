#coding=utf-8
#来自：深度学习进阶：算法与应用（麦子学院）
#根据平均灰度值判断数字所属类别
#支持字典的一个包
from collections import defaultdict

# My libraries
import mnist_loader

def main():
    #下载数据集
    training_data, validation_data, test_data = mnist_loader.load_data()
    #计算每一张图片的平均灰度值
    avgs = avg_darknesses(training_data)
    #猜测正确的图片，统计正确的数字
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))
    print "Baseline classifier using average darkness of image."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
#计算图片的平均灰度值
def avg_darknesses(training_data):
    #实际的每个数字包含的实例数
    digit_counts = defaultdict(int)
    #每个数字包含的实例对应图片的像素值和
    darknesses = defaultdict(float)
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1
        darknesses[digit] += sum(image)
    #print digit_counts
    #print darknesses
    avgs = defaultdict(float)
    #计算每个数字的平均灰度值
    for digit, n in digit_counts.iteritems():
        avgs[digit] = darknesses[digit] / n
    print avgs
    return avgs
#参数：图片，平均灰度值
def guess_digit(image, avgs):
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.iteritems()}
    #返回最接近的图片的k,即猜测的数值
    #print min(distances, key=distances.get)
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()
