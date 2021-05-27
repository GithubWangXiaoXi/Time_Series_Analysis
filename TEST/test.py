# 引入模块
import time, datetime

# [python中时间、日期、时间戳的转换](https://www.cnblogs.com/jfl-xx/p/8024596.html)
# 字符类型的时间
tss1 = '2013-10-10 23:40:00'
# 转为时间数组
timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
print(timeArray)
# timeArray可以调用tm_year等
print(timeArray.tm_year)   # 2013
# 转为时间戳
timeStamp = int(time.mktime(timeArray))
print(timeStamp)  # 1381419600

tss2 = "2013-10-10 23:40:00"
# 转为数组
timeArray = time.strptime(tss2, "%Y-%m-%d %H:%M:%S")
# 转为其它显示格式
otherStyleTime = time.strftime("%Y/%m/%d %H:%M:%S", timeArray)
print(otherStyleTime)  # 2013/10/10 23:40:00

tss3 = "2013/10/10 23:40:00"
timeArray = time.strptime(tss3, "%Y/%m/%d %H:%M:%S")
otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
print(otherStyleTime)  # 2013-10-10 23:40:00